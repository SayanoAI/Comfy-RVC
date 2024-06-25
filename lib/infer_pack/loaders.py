from collections import defaultdict
import os
from typing import Dict, List, Optional
import torch
from transformers import HubertModel, HubertConfig
import safetensors.torch as st
from safetensors import safe_open
import json

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)

    @staticmethod
    def from_safetensors(path: str, device="cpu", framework="pt"):
        assert path.endswith(".safetensors"), f"{path} must end with '.safetensors'"
        
        with safe_open(path, framework=framework, device="cpu") as f:
            metadata = f.metadata()
            state_dict = {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model = HubertModelWithFinalProj(HubertConfig.from_dict(json.loads(metadata["config"])))
        model.load_state_dict(state_dict=state_dict)
        model.eval()
        return model.to(device)

    def to_safetensors(self, sf_filename: str, discard_names: Optional[List[str]]=[]):
        assert hasattr(self, "state_dict"), f"Model does not have state_dict"

        state_dict = self.state_dict()
        to_removes = _remove_duplicate_names(state_dict, discard_names=discard_names)

        metadata = dict(format="pt", config=json.dumps(self.config.to_dict()))
        for kept_name, to_remove_group in to_removes.items():
            for to_remove in to_remove_group:
                if to_remove not in metadata:
                    metadata[to_remove] = kept_name
                del state_dict[to_remove]
        # Force tensors to be contiguous
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}

        dirname = os.path.dirname(sf_filename)
        os.makedirs(dirname, exist_ok=True)
        st.save_file(state_dict, sf_filename, metadata=metadata)
        reloaded = HubertModelWithFinalProj.from_safetensors(sf_filename, device=self.device)
        assert self.equals(reloaded), f"{sf_filename} does not equal source!"
        del reloaded

    def extract_features(self, source: torch.Tensor, version="v2", **kwargs):
        with torch.no_grad():
            output_layer = 9 if version == "v1" else 12
            # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/hubert/hubert.py#L476
            output = self(source.to(self.config.torch_dtype), output_hidden_states=True)["hidden_states"][output_layer-1]
            features = self.final_proj(output) if version == "v1" else output
        return features.to(source.dtype)
    
    def equals(self, other: "HubertModelWithFinalProj"):
        # check config
        if self.config.to_dict()!=other.config.to_dict(): return False

        # check parameters
        self_dict = self.state_dict()
        other_dict = other.state_dict()
        for k in other_dict:
            if not torch.equal(self_dict[k], other_dict[k]):
                print(f"{k} is not the same")
                return False
        # for p1, p2 in zip(self.parameters(), other.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0: return False
            
        # check output
        inputs = torch.ones((1,16000))
        if self(inputs).last_hidden_state.data.ne(other(inputs).last_hidden_state.data).sum() > 0: return False

        return True
    
# from: https://github.com/huggingface/safetensors/blob/main/bindings/python/convert.py#L36
def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = st._find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if st._is_complete(state_dict[name])])
        if not complete_names:
            if len(shared) == 1:
                # Force contiguous
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model since you could be storing much more memory than needed. Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an issue."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mecanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove
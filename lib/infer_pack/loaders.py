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
    def load_safetensors(path: str, device="cpu"):
        assert path.endswith(".safetensors"), f"{path} must end with '.safetensors'"
        
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            state_dict = {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model = HubertModelWithFinalProj(HubertConfig.from_dict(json.loads(metadata["config"])))
        model.load_state_dict(state_dict=state_dict)
        return model.to(device)
    
    def save_safetensors(self, path: str):
        assert path.endswith(".safetensors"), f"{path} must end with '.safetensors'"
        
        with open(path,"wb") as f:
            state_dict = self.state_dict()
            f.write(st.save(state_dict,dict(config=json.dumps(self.config.to_dict()))))

    def extract_features(self, source: torch.Tensor, version="v2", **kwargs):
        with torch.no_grad():
            output_layer = 9 if version == "v1" else 12
            output = self(source.to(self.config.torch_dtype), output_hidden_states=True)["hidden_states"][output_layer]
            features = self.final_proj(output) if version == "v1" else output
        return features.to(source.dtype)
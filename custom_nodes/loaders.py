import os
from .settings import PITCH_EXTRACTION_OPTIONS
from ..lib import BASE_MODELS_DIR
from ..lib.model_utils import load_hubert
from ..lib.utils import get_filenames, get_hash, get_optimal_torch_device
from .utils import model_downloader
import torch
import folder_paths
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ..vc_infer_pipeline import get_vc
from .settings.downloader import RVC_DOWNLOAD_LINK, RVC_INDEX, RVC_MODELS, download_file
from ..config import config

input_path = folder_paths.get_input_directory()
temp_path = folder_paths.get_temp_directory()
CATEGORY = "🌺RVC-Studio/loaders"

model_ids = [
    'openai/whisper-large-v3',
    'openai/whisper-large-v2',
    'openai/whisper-large',
    'openai/whisper-medium',
    'openai/whisper-small',
    'openai/whisper-base',
    'openai/whisper-tiny',
    'openai/whisper-medium.en',
    'openai/whisper-small.en',
    'openai/whisper-base.en',
    'openai/whisper-tiny.en',
]

languages = ['en', 'fr', 'es']

class LoadPitchExtractionParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'f0_method': (PITCH_EXTRACTION_OPTIONS,{"default": "rmvpe"}),
                "f0_autotune": ("BOOLEAN",),
                "index_rate": ("FLOAT",{
                    "default": .75, 
                    "min": 0., #Minimum value
                    "max": 1., #Maximum value
                    "step": .01, #Slider's step
                    "display": "slider"
                }),
                "resample_sr": ([0,16000,32000,40000,44100,48000],{"default": 0}),
                "rms_mix_rate": ("FLOAT",{
                    "default": 0.25, 
                    "min": 0., #Minimum value
                    "max": 1., #Maximum value
                    "step": .01, #Slider's step
                    "display": "slider"
                }),
                "protect": ("FLOAT",{
                    "default": 0.25, 
                    "min": 0., #Minimum value
                    "max": .5, #Maximum value
                    "step": .01, #Slider's step
                    "display": "slider"
                })
            },
        }

    RETURN_TYPES = ('PITCH_EXTRACTION', )
    RETURN_NAMES = ('pitch_extraction_params', )

    CATEGORY = CATEGORY

    FUNCTION = 'load_params'

    def load_params(self, **params):
        if "rmvpe" in params.get("f0_method",""): model_downloader("rmvpe.pt")
        return (params,)
    
    @classmethod
    def IS_CHANGED(cls, **params):
        return get_hash(**params)

class LoadHubertModel:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = ["content-vec-best.safetensors"] + get_filenames(root=BASE_MODELS_DIR,folder=".",exts=["pt","safetensors"],format_func=os.path.basename)
        model_list = list(set(model_list)) # dedupe
        
        return {
            'required': {
                'model': (model_list,{"default": "content-vec-best.safetensors"}),
            },
        }

    RETURN_TYPES = ('HUBERT_MODEL', )
    RETURN_NAMES = ('hubert_model', )

    CATEGORY = CATEGORY

    FUNCTION = 'load_model'

    def load_model(self, model):
        model_path = model_downloader(model)
        hubert_model = lambda:load_hubert(model_path,config=config)
        return (hubert_model,)
    
    @classmethod
    def IS_CHANGED(cls, model):
        return get_hash(model)
    
class LoadRVCModelNode:

    @classmethod
    def INPUT_TYPES(cls):
        model_list = RVC_MODELS + get_filenames(root=BASE_MODELS_DIR,folder="RVC",exts=["pth"],format_func=lambda x: f"RVC/{os.path.basename(x)}")
        model_list = list(set(model_list)) # dedupe
        index_list = ["None"] + RVC_INDEX + get_filenames(root=BASE_MODELS_DIR,folder="RVC",exts=["index"],format_func=lambda x: f"RVC/.index/{os.path.basename(x)}")
        index_list = list(set(index_list)) # dedupe

        return {
            'required': {
                'model': (model_list,{"default": model_list[0]}),
            },
            "optional": {
                "index": (index_list,{"default": "None"}),
            }
        }

    RETURN_TYPES = ('RVC_MODEL', 'STRING')
    RETURN_NAMES = ('model', 'model_name')

    CATEGORY = CATEGORY

    FUNCTION = 'load_model'


    def load_model(self, model, index="None"):
        model_path = file_index = None
        try:
            filename = os.path.basename(model)
            subfolder = os.path.dirname(model)
            model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
            
            if not os.path.isfile(model_path):
                download_link = f"{RVC_DOWNLOAD_LINK}{model}"
                if download_file((model_path, download_link)): print(f"successfully downloaded: {model_path}")

            if not index=="None":
                file_index = os.path.join(BASE_MODELS_DIR,subfolder,".index",os.path.basename(index))
                if not os.path.isfile(file_index):
                    download_link = f"{RVC_DOWNLOAD_LINK}{index}"
                    if download_file((file_index, download_link)): print(f"successfully downloaded: {file_index}")
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {e}")
            raise e
        finally: return (lambda:get_vc(model_path, file_index),filename.split(".")[0])

    @classmethod
    def IS_CHANGED(cls, model, index):
        return get_hash(model, index)
    
class LoadWhisperModelNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model_id': (model_ids,{"default": "openai/whisper-base.en"}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 128, "min": 16, "max": 1024, "display": "slider"}),
                "chunk_length_s": ("INT", {"default": 30, "min": 15, "max": 60, "display": "slider"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 128, "display": "slider"}),
            }
        }

    RETURN_TYPES = ('TRANSCRIPTION_MODEL', )
    RETURN_NAMES = ('model', )

    CATEGORY = CATEGORY

    FUNCTION = 'load_model'


    def load_model(self, model_id, max_new_tokens=128, chunk_length_s=12, batch_size=16):
        device = get_optimal_torch_device()
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)

        # generate_kwargs = {}

        def pipe(): return pipeline(
                'automatic-speech-recognition',
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=max_new_tokens,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
        return ([pipe,model_id], )
    
    @classmethod
    def IS_CHANGED(cls, model_id, max_new_tokens, chunk_length_s, batch_size):
        return get_hash(model_id, max_new_tokens, chunk_length_s, batch_size)


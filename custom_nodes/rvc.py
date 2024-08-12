import os
import shutil
from ..config import config

from ..lib.model_utils import load_hubert

from .utils import model_downloader
from .settings import PITCH_EXTRACTION_OPTIONS
from .settings.downloader import RVC_DOWNLOAD_LINK, RVC_INDEX, RVC_MODELS, download_file

from ..lib.audio import SUPPORTED_AUDIO, audio_to_bytes, bytes_to_audio, load_input_audio, save_input_audio

from ..vc_infer_pipeline import get_vc, vc_single
import folder_paths
from ..lib.utils import get_filenames, get_hash, get_optimal_torch_device
from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR

input_path = folder_paths.get_input_directory()
temp_path = folder_paths.get_temp_directory()
cache_dir = os.path.join(BASE_CACHE_DIR,"rvc")
# output_path = folder_paths.get_output_directory()
# base_path = os.path.dirname(input_path)
node_path = os.path.join(BASE_MODELS_DIR,"custom_nodes/ComfyUI-UVR5")
weights_path = os.path.join(BASE_MODELS_DIR, "uvr5")
device = get_optimal_torch_device()
CATEGORY = "🌺RVC-Studio/rvc"

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
    
class RVCNode:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):


        return {
            "required": {
                "audio": ("VHS_AUDIO",),
                "model": ("RVC_MODEL",),
                "hubert_model": ("HUBERT_MODEL",),
                "pitch_extraction_params": ("PITCH_EXTRACTION",),
                "f0_up_key": ("INT",{
                    "default": 0, 
                    "min": -14, #Minimum value
                    "max": 14, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider"
                }),
            },
            "optional": {
                "format":(SUPPORTED_AUDIO,{"default": "flac"}),
                "use_cache": ("BOOLEAN",{"default": True})
            }
        }
    
    OUTPUT_NODE = True

    RETURN_TYPES = ("VHS_AUDIO",)

    FUNCTION = "convert"

    CATEGORY = CATEGORY

    def convert(self, audio, model, hubert_model, pitch_extraction_params, f0_up_key, format="flac", use_cache=True):
        
        widgetId = get_hash(audio(), model, f0_up_key, *pitch_extraction_params.items())
        cache_name = os.path.join(BASE_CACHE_DIR,"rvc",f"{widgetId}.{format}")

        if use_cache and os.path.isfile(cache_name): output_audio = load_input_audio(cache_name)
        else:
            input_audio = bytes_to_audio(audio())
            output_audio = vc_single(hubert_model=hubert_model(),input_audio=input_audio,f0_up_key=f0_up_key,**model(),**pitch_extraction_params)
            
            if use_cache:
                print(save_input_audio(cache_name, output_audio))
                if os.path.isfile(cache_name): output_audio = load_input_audio(cache_name)
        
        tempdir = os.path.join(temp_path,"preview")
        os.makedirs(tempdir, exist_ok=True)
        audio_name = os.path.basename(cache_name)
        preview_file = os.path.join(tempdir,audio_name)
        if not os.path.isfile(preview_file): shutil.copyfile(cache_name,preview_file)
        return {"ui": {"preview": [{"filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId}]}, "result": (lambda:audio_to_bytes(*output_audio),)}

    


import os
import shutil

from ..lib.audio import SUPPORTED_AUDIO, audio_to_bytes, bytes_to_audio, load_input_audio, save_input_audio

from ..vc_infer_pipeline import vc_single
import folder_paths
from ..lib.utils import get_hash, get_optimal_torch_device
from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR

input_path = folder_paths.get_input_directory()
temp_path = folder_paths.get_temp_directory()
cache_dir = os.path.join(BASE_CACHE_DIR,"rvc")
# output_path = folder_paths.get_output_directory()
# base_path = os.path.dirname(input_path)
node_path = os.path.join(BASE_MODELS_DIR,"custom_nodes/ComfyUI-UVR5")
weights_path = os.path.join(BASE_MODELS_DIR, "uvr5")
device = get_optimal_torch_device()
is_half = True

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

    CATEGORY = "🌺RVC-Studio/rvc"

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


    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return get_hash(*args, *kwargs.items())
    


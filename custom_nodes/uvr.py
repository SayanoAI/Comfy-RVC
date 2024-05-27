import os

from ..lib.audio import audio_to_bytes, bytes_to_audio, save_input_audio
import folder_paths
from ..lib.utils import get_filenames, get_hash, get_optimal_torch_device
from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR, karafan
from ..uvr5_cli import Separator
from .settings.downloader import KARAFAN_MODELS, MDX_MODELS, RVC_DOWNLOAD_LINK, VR_MODELS, download_file

temp_path = folder_paths.get_temp_directory()
cache_dir = os.path.join(BASE_CACHE_DIR,"uvr")
device = get_optimal_torch_device()
is_half = True

class UVR5Node:
 
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):

        model_list = MDX_MODELS + VR_MODELS + KARAFAN_MODELS + get_filenames(root=BASE_MODELS_DIR,format_func=lambda x: f"{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}",name_filters=["UVR","MDXNET","karafan"])
        model_list = list(set(model_list)) # dedupe

        return {
            "required": {
                "audio": ("VHS_AUDIO",),
                "model": (model_list,{
                    "default": "UVR/HP5-vocals+instrumentals.pth"
                }),
                "agg":("INT",{
                    "default": 10, 
                    "min": 0, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider"
                }),
                "format":(["wav", "flac", "mp3"],{
                    "default": "flac"
                }),
            },
            "optional": {
                "use_cache": ("BOOLEAN",{"default": True})
            }
        }

    RETURN_TYPES = ("VHS_AUDIO","VHS_AUDIO","VHS_AUDIO")
    RETURN_NAMES = ("primary_stem","secondary_stem","audio_passthrough")

    FUNCTION = "split"

    CATEGORY = "🌺RVC-Studio/uvr"

    def split(self, audio, model, **kwargs):
        filename = os.path.basename(model)
        subfolder = os.path.dirname(model)
        model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
        if not os.path.isfile(model_path):
            download_link = f"{RVC_DOWNLOAD_LINK}{model}"
            params = model_path, download_link
            if download_file(params): print(f"successfully downloaded: {model_path}")
        
        input_audio = bytes_to_audio(audio())
        audio_path = os.path.join(temp_path,"uvr",f"{get_hash(audio(),model, *kwargs.items())}.wav")
        if not os.path.isfile(audio_path):
            os.makedirs(os.path.dirname(audio_path),exist_ok=True)
            print(save_input_audio(audio_path,input_audio))

        if "karafan" in model_path:
            vocals, instrumental, input_audio = karafan.inference.Process(audio_path,cache_dir=cache_dir,**kwargs)
        else:
            model = Separator(
                    model_path=model_path,
                    device=device,
                    is_half="cuda" in str(device),
                    cache_dir=cache_dir,              
                    **kwargs
                    )
            vocals, instrumental, input_audio = model.run_inference(audio_path,format=kwargs.get("format"))
        
        return (lambda:audio_to_bytes(*vocals), lambda:audio_to_bytes(*instrumental), audio)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        print(f"{args=} {kwargs=}")
        return get_hash(*args, *kwargs.items())
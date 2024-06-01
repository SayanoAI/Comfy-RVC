import os

from ..lib.audio import audio_to_bytes, bytes_to_audio, save_input_audio, load_input_audio
import folder_paths
from ..lib.utils import get_filenames, get_hash, get_file_hash, get_optimal_torch_device
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

        model_list = MDX_MODELS + VR_MODELS + KARAFAN_MODELS + get_filenames(root=BASE_MODELS_DIR,format_func=lambda x: f"{os.path.basename(os.path.dirname(x))}/{os.path.basename(x)}",name_filters=["UVR","MDX","karafan"])
        model_list = list(set(model_list)) # dedupe

        return {
            "required": {
                "audio": ("VHS_AUDIO",),
                "model": (model_list,{
                    "default": "UVR/HP5-vocals+instrumentals.pth"
                }),
            },
            "optional": {
                "use_cache": ("BOOLEAN",{"default": True}),
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
            }
        }

    RETURN_TYPES = ("VHS_AUDIO","VHS_AUDIO","VHS_AUDIO")
    RETURN_NAMES = ("primary_stem","secondary_stem","audio_passthrough")

    FUNCTION = "split"

    CATEGORY = "🌺RVC-Studio/uvr"

    def split(self, audio, model, use_cache=True, agg=10, format='flac'):
        filename = os.path.basename(model)
        subfolder = os.path.dirname(model)
        model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
        if not os.path.isfile(model_path):
            download_link = f"{RVC_DOWNLOAD_LINK}{model}"
            params = model_path, download_link
            if download_file(params): print(f"successfully downloaded: {model_path}")
        
        input_audio = bytes_to_audio(audio())
        hash_name = get_hash(audio(), model, agg, format)
        audio_path = os.path.join(temp_path,"uvr",f"{hash_name}.wav")
        vocal_path = os.path.join(cache_dir,hash_name,f"primary.{format}")
        instrumental_path = os.path.join(cache_dir,hash_name,f"secondary.{format}")
        if os.path.isfile(vocal_path) and os.path.isfile(instrumental_path) and use_cache:
            vocals = load_input_audio(vocal_path)
            instrumental = load_input_audio(instrumental_path)
        else:
            if not os.path.isfile(audio_path):
                os.makedirs(os.path.dirname(audio_path),exist_ok=True)
                print(save_input_audio(audio_path,input_audio))
            vocals=instrumental=None

            try: # try original RVC implementation
                model = Separator(
                        model_path=model_path,
                        device=device,
                        is_half="cuda" in str(device),
                        cache_dir=cache_dir,
                        agg=agg
                        )
                vocals, instrumental, input_audio = model.run_inference(audio_path,format=format)
            except Exception as e:
                print(f"Error: {e}")
                if "karafan" in model_path: # try karafan implementation
                    vocals, instrumental, input_audio = karafan.inference.Process(audio_path,cache_dir=temp_path,format=format)
                else: # try python-audio-separator implementation
                    import audio_separator.separator as uvr
                    model_dir = os.path.dirname(model_path)
                    model_name = os.path.basename(model_path)
                    vr_params={"batch_size": 4, "window_size": 512, "aggression": agg, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": "mirroring"}
                    model = uvr.Separator(model_file_dir=os.path.join(BASE_MODELS_DIR,model_dir),output_dir=temp_path,output_format=format,vr_params=vr_params)
                    model.load_model(model_name)
                    output_files = model.separate(audio_path)
                    print(f"{output_files=}")
                    vocals = load_input_audio(os.path.join(temp_path,output_files[0]))
                    instrumental = load_input_audio(os.path.join(temp_path,output_files[1]))
            finally:
                if vocals is not None and instrumental is not None and use_cache:
                    print(save_input_audio(vocal_path,vocals))
                    print(save_input_audio(instrumental_path,instrumental))
        
        return (lambda:audio_to_bytes(*vocals), lambda:audio_to_bytes(*instrumental), audio)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        print(f"{args=} {kwargs=}")
        return get_hash(args,kwargs)
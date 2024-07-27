import os
import audio_separator.separator as uvr
from ..lib.audio import audio_to_bytes, bytes_to_audio, save_input_audio, load_input_audio
import folder_paths
from ..lib.utils import get_filenames, get_hash, get_optimal_torch_device
from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR, karafan
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

    RETURN_TYPES = ("VHS_AUDIO","VHS_AUDIO")
    RETURN_NAMES = ("primary_stem","secondary_stem")

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
        primary_path = os.path.join(cache_dir,hash_name,f"primary.{format}")
        secondary_path = os.path.join(cache_dir,hash_name,f"secondary.{format}")
        primary=secondary=None

        if os.path.isfile(primary_path) and os.path.isfile(secondary_path) and use_cache:
            primary = load_input_audio(primary_path)
            secondary = load_input_audio(secondary_path)
        else:
            if not os.path.isfile(audio_path):
                os.makedirs(os.path.dirname(audio_path),exist_ok=True)
                print(save_input_audio(audio_path,input_audio))
            
            try: 
                if "karafan" in model_path: # try karafan implementation
                    primary, secondary, _ = karafan.inference.Process(audio_path,cache_dir=temp_path,format=format)
                else: # try python-audio-separator implementation
                    model_dir = os.path.dirname(model_path)
                    model_name = os.path.basename(model_path)
                    vr_params={"batch_size": 4, "window_size": 512, "aggression": agg, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": "mirroring"}
                    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 4}
                    model = uvr.Separator(model_file_dir=os.path.join(BASE_MODELS_DIR,model_dir),output_dir=temp_path,output_format=format,vr_params=vr_params,mdx_params=mdx_params)
                    model.load_model(model_name)
                    output_files = model.separate(audio_path)
                    primary = load_input_audio(os.path.join(temp_path,output_files[0]))
                    secondary = load_input_audio(os.path.join(temp_path,output_files[1]))
            except Exception as e: # try RVC implementation
                print(f"Error: {e}")
                
                from ..uvr5_cli import Separator
                model = Separator(
                    model_path=model_path,
                    device=device,
                    is_half="cuda" in str(device),
                    cache_dir=cache_dir,
                    agg=agg
                    )
                primary, secondary, _ = model.run_inference(audio_path,format=format)
            finally:
                if primary is not None and secondary is not None and use_cache:
                    print(save_input_audio(primary_path,primary))
                    print(save_input_audio(secondary_path,secondary))

                if os.path.isfile(primary_path) and os.path.isfile(secondary_path) and use_cache:
                    primary = load_input_audio(primary_path)
                    secondary = load_input_audio(secondary_path)
        
        return (lambda:audio_to_bytes(*primary), lambda:audio_to_bytes(*secondary))

    # @classmethod
    # def IS_CHANGED(cls, *args, **kwargs):
    #     return get_hash(*args,*kwargs.items())
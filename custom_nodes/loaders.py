from io import BytesIO
import os
from pytube import YouTube
from .settings import PITCH_EXTRACTION_OPTIONS
from ..lib import BASE_MODELS_DIR
from ..lib.model_utils import load_hubert
from ..lib.utils import get_file_hash, get_filenames, get_hash, get_optimal_torch_device
from .utils import model_downloader
import torch
import folder_paths
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ..vc_infer_pipeline import get_vc
from .settings.downloader import RVC_DOWNLOAD_LINK, RVC_INDEX, RVC_MODELS, download_file
from ..lib.audio import SUPPORTED_AUDIO, audio_to_bytes, load_input_audio
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

    @classmethod
    def IS_CHANGED(cls, model):
        return get_hash(model)

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

    @classmethod
    def IS_CHANGED(cls, model):
        return get_hash(model)

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
                # generate_kwargs=generate_kwargs,
            )
        return ([pipe,model_id], )

class LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = input_path
        files = get_filenames(root=input_dir,exts=SUPPORTED_AUDIO,format_func=os.path.basename)
        
        return {
            "required": {
                "audio": (files,),
                "sr": (["None",16000,44100,48000],{"default": "None"}),
            }}

    CATEGORY = CATEGORY

    RETURN_TYPES = ("STRING","VHS_AUDIO")
    RETURN_NAMES = ("audio_name","vhs_audio")
    FUNCTION = "load_audio"

    def load_audio(self, audio, sr):
        audio_path = os.path.join(input_path,audio) #folder_paths.get_annotated_filepath(audio)
        widgetId = get_hash(audio_path)
        audio_name = os.path.basename(audio).split(".")[0]
        sr = None if sr=="None" else int(sr)
        audio = load_input_audio(audio_path,sr=sr)
        return {"ui": {"preview": [{"filename": audio_name, "type": "input", "widgetId": widgetId}]}, "result": (audio_name, lambda:audio_to_bytes(*audio))}

    @classmethod
    def IS_CHANGED(cls, audio):
        audio_path = os.path.join(input_path,audio)
        print(f"{audio_path=}")
        return get_file_hash(audio_path)
    
class DownloadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "url": ("STRING", {"default": ""})
            },
            "optional": {
                "sr": (["None",16000,44100,48000],{"default": "None"}),
                "song_name": ("STRING",{"default": ""},)
            }
        }

    CATEGORY = CATEGORY

    RETURN_TYPES = ("STRING","VHS_AUDIO")
    RETURN_NAMES = ("audio_name","vhs_audio")
    FUNCTION = "download_audio"

    def download_audio(self, url, sr="None", song_name=""):

        assert "youtube" in url, "Please provide a valid youtube URL!"
        widgetId = get_hash(url, sr)
        sr = None if sr=="None" else int(sr)
        audio_name = widgetId if song_name=="" else song_name
        audio_path = os.path.join(input_path,f"{audio_name}.mp3")

        if os.path.isfile(audio_path): input_audio = load_input_audio(audio_path,sr=sr)
        else:
            youtube_video = YouTube(url)
            audio = youtube_video.streams.get_audio_only(subtype="mp4")
            buffer = BytesIO()
            audio.stream_to_buffer(buffer)
            buffer.seek(0)
            with open(audio_path,"wb") as f:
                f.write(buffer.read())
            buffer.close()
            input_audio = load_input_audio(audio_path,sr=sr)
            del buffer, audio

        return {"ui": {"preview": [{"filename": os.path.basename(audio_path), "type": "input", "widgetId": widgetId}]}, "result": (audio_name, lambda:audio_to_bytes(*input_audio))}

    @classmethod
    def IS_CHANGED(cls, url):
        return get_hash(url)

from io import BytesIO
import os
import shutil
import numpy as np
from pytube import YouTube
import torch
from .settings import MERGE_OPTIONS
from .utils import increment_filename_no_overwrite
from ..lib.audio import MAX_INT16, SUPPORTED_AUDIO, audio_to_bytes, bytes_to_audio, load_input_audio, pad_audio, remix_audio, save_input_audio
from ..lib.utils import get_filenames, get_hash, get_merge_func
import folder_paths

CATEGORY = "🌺RVC-Studio/audio"
input_path = folder_paths.get_input_directory()
temp_path = folder_paths.get_temp_directory()

def to_audio_dict(audio, sr):
    #from https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/bf2a9402d0b2727c7170c43621d569f4d531015f/videohelpersuite/nodes.py#L706C22-L706C67
    waveform = torch.from_numpy(audio).reshape((-1,audio.ndim)).transpose(0,1).unsqueeze(0) 
    print(f"{waveform.shape=} {waveform.ndim=} {audio.shape=} {audio.ndim=}")
    return dict(waveform=waveform,sample_rate=sr)


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

    RETURN_TYPES = ("STRING","VHS_AUDIO","AUDIO")
    RETURN_NAMES = ("audio_name","vhs_audio","audio")
    FUNCTION = "load_audio"

    def load_audio(self, audio, sr):
        audio_path = os.path.join(input_path,audio) #folder_paths.get_annotated_filepath(audio)
        widgetId = get_hash(audio_path)
        audio_name = os.path.basename(audio).split(".")[0]
        sr = None if sr=="None" else int(sr)
        audio = load_input_audio(audio_path,sr=sr)

        return {"ui": {"preview": [{"filename": audio_name, "type": "input", "widgetId": widgetId}]}, "result": (audio_name, lambda:audio_to_bytes(*audio), to_audio_dict(*audio))}
    
    @classmethod
    def IS_CHANGED(cls, audio, sr):
        return get_hash(audio,sr)

    
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

    RETURN_TYPES = ("STRING","VHS_AUDIO","AUDIO")
    RETURN_NAMES = ("audio_name","vhs_audio","audio")
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

        return {"ui": {"preview": [{"filename": os.path.basename(audio_path), "type": "input", "widgetId": widgetId}]}, "result": (audio_name, lambda:audio_to_bytes(*input_audio),to_audio_dict(*audio))}

    @classmethod
    def IS_CHANGED(cls, url, sr, song_name):
        return get_hash(url,sr,song_name)
    
class MergeAudioNode:
   
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("VHS_AUDIO",),
                "audio2": ("VHS_AUDIO",),
            },
            "optional": {
                "sr": (["None",32000,40000,44100,48000],{
                    "default": "None"
                }),
                "merge_type": (MERGE_OPTIONS,{"default": "median"}),
                "normalize": ("BOOLEAN",{"default": True}),
                "audio3_opt": ("VHS_AUDIO",{"default": None}),
                "audio4_opt": ("VHS_AUDIO",{"default": None}),
            }
        }

    RETURN_TYPES = ("VHS_AUDIO","AUDIO")
    RETURN_NAMES = ("vhs_audio","audio")

    FUNCTION = "merge"

    CATEGORY = CATEGORY

    def merge(self, audio1, audio2, sr="None", merge_type="median", normalize=False, audio3_opt=None, audio4_opt=None):

        audios = [audio() for audio in [audio1, audio2, audio3_opt, audio4_opt] if audio is not None]
        widgetId = get_hash(*audios,sr,merge_type,normalize)
        audio_path = os.path.join(temp_path,"preview",f"{widgetId}.flac")

        if os.path.isfile(audio_path): merged_audio = load_input_audio(audio_path)
        else:
            input_audios = [bytes_to_audio(audio) for audio in audios]
            merged_sr = min([sr for (_,sr) in input_audios]) if sr=="None" else sr
            input_audios = [remix_audio(audio,merged_sr,norm=normalize) for audio in input_audios]
            merge_func = get_merge_func(merge_type)
            merged_audio = merge_func(pad_audio(*[audio for (audio,_) in input_audios],axis=0),axis=0), merged_sr
            print(save_input_audio(audio_path,merged_audio))
            del input_audios
            if os.path.isfile(audio_path): merged_audio = load_input_audio(audio_path)
            
        del audios
        audio_name = os.path.basename(audio_path)
        return {"ui": {"preview": [{"filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId}]}, "result": (lambda: audio_to_bytes(*merged_audio),to_audio_dict(*merged_audio))}
    
    @classmethod
    def IS_CHANGED(cls, audio1, audio2, sr, merge_type, normalize, audio3_opt=None, audio4_opt=None):
        audios = [audio() for audio in [audio1, audio2, audio3_opt, audio4_opt] if audio is not None]
        return get_hash(sr, merge_type, normalize, *audios)
    
class PreviewAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("VHS_AUDIO",),
                "filename": ("STRING",{"default": "test"}),
                "save_format": (SUPPORTED_AUDIO,{"default": "flac"},),
                "save_channels": ([1,2],{"default": 1}),
                "overwrite_existing": ("BOOLEAN", {"default": True}),
                "autoplay": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING","VHS_AUDIO","AUDIO")
    RETURN_NAMES = ("output_path","vhs_audio","audio")

    OUTPUT_NODE = True

    CATEGORY = CATEGORY

    FUNCTION = "save_audio"

    def save_audio(self, audio, filename, save_format, save_channels, overwrite_existing, autoplay):

        filename = filename.strip()
        assert filename, "Filename cannot be empty"

        base_output_dir = folder_paths.get_output_directory()
        assert os.path.exists(base_output_dir), f"Output directory {base_output_dir} does not exist"
        output_dir = os.path.join(base_output_dir, 'audio')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.{save_format}")

        if os.path.isfile(output_path) and not overwrite_existing:
            output_path = increment_filename_no_overwrite(output_path)
        
        input_audio = bytes_to_audio(audio())
        print(save_input_audio(output_path,input_audio,to_int16=True,to_stereo=save_channels==2))

        tempdir = os.path.join(temp_path,"preview")
        os.makedirs(tempdir, exist_ok=True)
        widgetId = get_hash(output_path,save_channels)
        audio_name = f"{widgetId}.{save_format}"
        preview_file = os.path.join(tempdir,audio_name)
        shutil.copyfile(output_path,preview_file)
        return {"ui": {"preview": [{
            "filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId, "autoplay": autoplay
            }]}, "result": (output_path, lambda:audio_to_bytes(*input_audio), to_audio_dict(*input_audio))}
    
class AudioBatchValueNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ('VHS_AUDIO',),
                "num_segments": ('INT', {"min": 2, "max": 128, "step": 1, "forceInput": True}),
                "output_min": ('FLOAT', {'default': 0., "min": -1000., "max": 1000., "step": .01}),
                "output_max": ('FLOAT', {'default': 1., "min": 0., "max": 1000., "step": .01}),
                "norm": (["scale","tanh","sigmoid"], {"default": "scale"}),
            },
            "optional": {
                "silence_threshold": ("INT", {"default": 1000, "min": 1, "max": MAX_INT16, "step": 1, "display": "slider"}),
                "duration_list": ("INT", {"default": None, "min": 1, "forceInput": True}),
                "print_output": ("BOOLEAN", {"default": False}),
                "inverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT","INT","INT")
    RETURN_NAMES = ("FLOAT","INT","num_values")

    FUNCTION = "get_frame_weights"

    CATEGORY = CATEGORY

    @staticmethod
    def get_rms(audio): # root mean squared of audio segment
        return np.sqrt(np.nanmean(audio**2))

    def get_frame_weights(self, audio, num_segments, output_min, output_max, norm,
                          silence_threshold=1000, duration_list=None, print_output=False, inverse=False):
        assert output_max>=output_min, f"{output_max=} must be greater or equal to {output_min=}!"

        audio_data = bytes_to_audio(audio())
        audio,_ = remix_audio(audio_data,norm=True,to_int16=True)
        num_values = int(num_segments)
        audio_rms = np.nan_to_num(list(map(self.get_rms,np.array_split(audio.flatten()/silence_threshold, num_values))),nan=0)
        audio_zscore = (audio_rms-audio_rms.mean())/audio_rms.std()
        output_range = output_max-output_min

        if norm=="tanh":
            x_norm = np.tanh(audio_zscore) #tanh=-1 to 1
            if inverse: x_norm*=-1
            x_norm = (x_norm * output_range + output_max + output_min)/2
        elif norm=="sigmoid":
            x_norm = 1. / (1. + np.exp(-audio_zscore)) #sigmoid=0 to 1
            if inverse: x_norm=1-x_norm
            x_norm = x_norm * output_range + output_min 
        else:
            x_min = audio_zscore.min() 
            x_norm = (audio_zscore - x_min) / (audio_zscore.max() - x_min) #scale=0 to 1
            if inverse: x_norm=1-x_norm
            x_norm = x_norm * output_range + output_min 

        # batch_value = ",\n".join([f'{n}:({v})' for n,v in enumerate(x_norm)])
        if print_output:
            print(f"{audio_rms.min()=} {audio_rms.max()=} {audio_rms.mean()=} {len(audio_rms)=}")
            print(f"{x_norm.min()=} {x_norm.max()=} {x_norm.mean()=} {len(x_norm)=}")
        
        if duration_list is not None:
            segments = np.cumsum(duration_list)
            x_norm = np.array_split(x_norm,segments)
            x_norm = map(list,x_norm)
            x_norm_int = [list(map(int,norms)) for norms in x_norm]
        else:
            x_norm_int = map(int,x_norm)

        return (list(x_norm),list(x_norm_int),num_values)
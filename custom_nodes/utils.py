import itertools
import os
from pprint import pprint
import re
import tempfile
import torch
import folder_paths
from .settings import MERGE_OPTIONS
from ..lib.utils import gc_collect, get_hash
from .settings.downloader import RVC_DOWNLOAD_LINK, download_file
from ..lib import BASE_MODELS_DIR
import numpy as np
from ..lib.audio import MAX_INT16, audio_to_bytes, bytes_to_audio, load_input_audio, merge_audio, remix_audio, save_input_audio

CATEGORY = "🌺RVC-Studio/utils"
temp_path = folder_paths.get_temp_directory()

def model_downloader(model):
    filename = os.path.basename(model)
    subfolder = os.path.dirname(model)
    model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
    if not os.path.isfile(model_path):
        download_link = f"{RVC_DOWNLOAD_LINK}{model}"
        params = model_path, download_link
        if download_file(params): print(f"successfully downloaded: {model_path}")
    return model_path

def increment_filename_no_overwrite(proposed_path):
    output_dir, filename_ext = os.path.split(proposed_path)
    filename, file_format = os.splitext(filename_ext)
    files = os.listdir(output_dir)
    files = filter(lambda f: os.path.isfile(os.path.join(output_dir, f)), files)
    files = filter(lambda f: f.startswith(filename), files)
    files = filter(lambda f: f.endswith('.' + file_format), files)
    file_numbers = [re.search(r'_(\d+)\.', f) for f in files]
    file_numbers = [int(f.group(1)) for f in file_numbers if f]
    this_file_number = max(file_numbers) + 1 if file_numbers else 1
    output_path = os.path.join(output_dir, filename + f'_{this_file_number}.' + file_format)
    return output_path

class MergeImageBatches:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "iterate": ("BOOLEAN",{"default": False})
            },
        }

    INPUT_IS_LIST = (True, False)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"

    CATEGORY = CATEGORY

    __archive__ = None

    @staticmethod
    def iter_images(fname,images):
        size = len(images)
        if size==0: return images
        shape = images[0].shape
        if len(shape)>3: shape=shape[1:]

        memmap = np.memmap(fname, mode='w+', dtype=np.float32, shape=(size,*shape))
        
        for i,img in enumerate(itertools.chain(*images)):
            if i >= len(memmap): #need to expand memmap
                print(f"index {i} is larger than {len(memmap)}. expanding memmap to {size+i}...")
                memmap.flush()
                memmap = np.memmap(fname, dtype=np.float32, mode='r+',shape=(size+i,*shape), order='F')
            memmap[i]=img
            # del img
            # gc_collect()
        memmap.flush()

        print(f"{fname=} size={i} shape={memmap.shape}")

        # del images
        images = torch.from_numpy(memmap[:i])
        
        return images
    
    def __del__(self):
        try:
            if os.path.exists(self.__archive__): os.remove(self.__archive__)
            gc_collect()
        except Exception as error:
            print(f"Failed to delete {self.__archive__}: {error=}")

    def merge(self, images, iterate):
        
        if len(images) <= 1:
            return (images[0],)
        else:
            if hasattr(iterate,"pop"): iterate=iterate.pop()
            print(f"{iterate=}")

            if iterate:
                if self.__archive__ is None:
                    with tempfile.NamedTemporaryFile(delete=False) as ntf:
                        self.__archive__ = ntf.name
                images = self.iter_images(self.__archive__,images)
            else:
                images = torch.cat(images)
            
            gc_collect()
            print(f"Merged images: {len(images)=} {images.shape=}")
            
            return (images,)
        
        
class MergeLatentBatches:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents": ("LATENT",),
        }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "merge"

    CATEGORY = CATEGORY

    @staticmethod
    def merge_tensor(tensors):
        return torch.cat(tensors, dim=0)

    def merge(self, latents):
        merged_latents = {}
        gc_collect()
        merged_latents["samples"] = self.merge_tensor([latent["samples"] for latent in latents])
        merged_latents["noise_mask"] = self.merge_tensor([latent["noise_mask"] for latent in latents])
        merged_latents["batch_index"] = self.merge_tensor([latent["batch_index"] for latent in latents])
        
        print(f'Merged latents: {len(merged_latents["samples"])=} {latents["batch_index"]=}')
        return (merged_latents,)
        
class AudioBatchValueNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ('VHS_AUDIO',),
                "num_frames": ('INT', {"min": 1, "max": 128, "step": 1}),
                "output_min": ('FLOAT', {'default': 0., "min": -1000., "max": 1000., "step": .01}),
                "output_max": ('FLOAT', {'default': 1., "min": 0., "max": 1000., "step": .01}),
                "norm": (["scale","tanh","sigmoid"], {"default": "scale"}),
            },
            "optional": {
                "silence_threshold": ("INT", {"default": 1000, "min": 1, "max": MAX_INT16, "step": 1, "display": "slider"}),
                "frame_multiplier": ("INT", {"default": 1, "min": 1, "max": 120, "step": 1}),
                "print_output": ("BOOLEAN", {"default": False}),
                "inverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING","INT")
    RETURN_NAMES = ("batch_value_text","num_values")

    FUNCTION = "get_frame_weights"

    CATEGORY = CATEGORY

    @staticmethod
    def get_rms(audio): # root mean squared of audio segment
        return np.sqrt(np.nanmean(audio**2))

    def get_frame_weights(self, audio, num_frames, output_min, output_max, norm,
                          silence_threshold=1000, frame_multiplier=1, print_output=False, inverse=False):
        assert output_max>=output_min, f"{output_max=} must be greater or equal to {output_min=}!"

        audio_data = bytes_to_audio(audio())
        audio,_ = remix_audio(audio_data,norm=True,to_int16=True)
        num_values = int(num_frames*frame_multiplier)
        audio_rms = np.nan_to_num(list(map(self.get_rms,np.array_split(audio.flatten()/silence_threshold, num_values))),nan=0)
        audio_zscore = (audio_rms-audio_rms.mean())/audio_rms.std()
        output_range = output_max-output_min

        print(f"{audio_rms.min()=} {audio_rms.max()=} {audio_rms.mean()=}")
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
        print(f"{x_norm.min()=} {x_norm.max()=} {x_norm.mean()=}")

        batch_value = ",\n".join([f'{n}:({v})' for n,v in enumerate(x_norm)])
        if print_output: pprint(batch_value)
        return (batch_value,num_values)
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        print(f"{args=} {kwargs=}")
        return get_hash(*args, *kwargs.items())
    
class ImageRepeatInterleavedNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "fps": ("INT", {"default": 1, "min": 1})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    FUNCTION = "rebatch"

    CATEGORY = CATEGORY

    def rebatch(self, images, repeats, fps=1):
        repeats = np.array(repeats).flatten()*fps
        print(f"{repeats=}")
        all_images = []
        for img in images:
            for i in range(img.shape[0]):
                all_images.append(img[i:i+1])

        # expand repeat
        if len(repeats)==1:
            repeats = repeats.repeat(len(all_images),axis=0)
        elif len(repeats)<len(all_images):
            pad_width = len(all_images) - len(repeats)
            repeats = np.pad(repeats, (0, pad_width), mode='constant', constant_values=1)

        for i,img in enumerate(all_images):
            if repeats[i]>1: all_images[i] = img.expand(repeats[i], *img.shape).flatten(0,1)
            else: all_images[i] = img
            print(f"{img.shape=} {all_images[i].shape=}")

        print(f"{len(images)=} {len(all_images)=}")
        gc_collect()

        return (all_images,)
    
class LatentRepeatInterleavedNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "fps": ("INT", {"default": 1, "min": 1})
            }
        }
    RETURN_TYPES = ("LATENT",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    FUNCTION = "merge"

    CATEGORY = CATEGORY

    @staticmethod
    def repeat(latents, repeat):
        if latents.ndim>3: #batched
            expanded_latents=(latents.expand(repeat, *latents.shape).flatten(0,1))
        else:
            expanded_latents=(latents.expand(repeat, *latents.shape))

        print(f"{len(latents)}*{repeat}={len(expanded_latents)} ({latents.shape=} => {expanded_latents.shape=})")
        return expanded_latents
    
    @staticmethod
    def rebatch(samples, repeats):
        offset = 0
        for i in range(len(samples)):

            samples[i]["batch_index"] = [i+offset for i in range(repeats[i])]
            offset += repeats[i]
            
            samples[i]["samples"] = LatentRepeatInterleavedNode.repeat(samples[i]["samples"],repeats[i])

            if "noise_mask" in samples[i]:
                samples[i]["noise_mask"] = LatentRepeatInterleavedNode.repeat(samples[i]["noise_mask"],repeats[i])

        return samples

    def merge(self, latents, repeats, fps=1):
        repeats = np.array(repeats).flatten()*fps
        print(f"{repeats=}")

        # combine all latents
        new_latents = []
        for latent in latents:
            if latent["samples"].ndim==3: #unbatched
                new_latents.append(latent)
            else:
                for i in range(len(latent["samples"])):
                    new_latent = {
                        "samples": latent["samples"][i],
                    }
                    if "noise_mask" in latent: new_latent["noise_mask"] = latent["noise_mask"][i]
                    new_latents.append(new_latent)

        # expand repeat
        if len(repeats)==1:
            repeats = repeats.repeat(len(new_latents),axis=0)
        elif len(repeats)<len(new_latents):
            pad_width = len(new_latents) - len(repeats)
            repeats = np.pad(repeats, (0, pad_width), mode='constant', constant_values=1)

        new_latents = self.rebatch(new_latents,repeats.astype(int))
        gc_collect()

        return (new_latents,)
    
class MergeAudioNode:
   
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("VHS_AUDIO",),
                "audio2": ("VHS_AUDIO",),
                "sr": (["None",32000,40000,44100,48000],{
                    "default": "None"
                }),
                "merge_type": (MERGE_OPTIONS,{"default": "mean"}),
                "normalize": ("BOOLEAN",{"default": True})
            }
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("vhs_audio",)

    FUNCTION = "merge"

    CATEGORY = CATEGORY

    def merge(self, audio1, audio2, sr, merge_type, normalize):
        if sr=="None": sr=None

        widgetId = get_hash(audio1(),audio2(),sr,merge_type,normalize)
        audio_path = os.path.join(temp_path,"preview",f"{widgetId}.flac")

        if os.path.isfile(audio_path): merged_audio = load_input_audio(audio_path, sr, norm=normalize)
        else:
            input_audio1 = bytes_to_audio(audio1())
            input_audio2 = bytes_to_audio(audio2())
            merged_audio = merge_audio(input_audio1,input_audio2,sr=sr,to_int16=True,norm=normalize,merge_type=merge_type)
            print(save_input_audio(audio_path,merged_audio))
        audio_name = os.path.basename(audio_path)
        return {"ui": {"preview": [{"filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId}]}, "result": (lambda: audio_to_bytes(*merged_audio),)}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        print(f"{args=} {kwargs=}")
        return get_hash(*args, *kwargs.items())
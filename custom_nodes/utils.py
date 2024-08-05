import itertools
import math
import os
from pprint import pprint
import re
import tempfile
import torch
import folder_paths
from .settings import MERGE_OPTIONS
from ..lib.utils import gc_collect, get_hash, get_merge_func
from .settings.downloader import RVC_DOWNLOAD_LINK, download_file
from ..lib import BASE_MODELS_DIR
import numpy as np
from ..lib.audio import MAX_INT16, audio_to_bytes, bytes_to_audio, load_input_audio, merge_audio, pad_audio, remix_audio, save_input_audio

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
                "iterate": ("BOOLEAN",{"default": True})
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
                image = images[0]
                for new_img in images[1:]:
                    image = torch.cat([image,new_img])
                    gc_collect()

                # if self.__archive__ is None:
                #     with tempfile.NamedTemporaryFile(delete=False) as ntf:
                #         self.__archive__ = ntf.name
                # images = self.iter_images(self.__archive__,images)
                del images
                images = image
                gc_collect()
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
        try:    
            if len(tensors): return torch.cat(tensors, dim=0)
            else: print(tensors)
        except Exception as error:
            print(f"{error=}")
        return []

    def merge(self, latents):
        merged_latent = {}
        gc_collect()
        merged_latent["samples"] = self.merge_tensor([latent["samples"] for latent in latents])
        merged_latent["noise_mask"] = self.merge_tensor([latent["noise_mask"] for latent in latents if "noise_mask" in latent])
        # if len(merged_latent["noise_mask"])>0: merged_latent["noise_mask"] = torch.mean(merged_latent["noise_mask"],dim=0)
        # merged_latent["batch_index"] = self.merge_tensor([latent["batch_index"] for latent in latents if "batch_index" in latent])
        merged_latent["batch_index"] = range(len(merged_latent["samples"]))
        
        # remove empty fields
        if merged_latent["noise_mask"]==[]: del merged_latent["noise_mask"]

        print(f'Merged latents: {len(merged_latent["samples"])=} {merged_latent["batch_index"]=}')
        return (merged_latent,)
        
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
        repeats = np.array(repeats).flatten()
        fps = np.array(fps)[0]
        print(f"{repeats=} {fps=}")
        all_images = []
        for img in images:
            for i in range(img.shape[0]):
                all_images.append(img[i:i+1])

        if len(repeats)==1:
            all_images*=repeats[0]
            repeats = np.ones(len(all_images))
        elif len(repeats)<len(all_images): # expand repeat
            pad_width = len(all_images) - len(repeats)
            repeats = np.pad(repeats, (0, pad_width), mode='constant', constant_values=1)

        for i,img in enumerate(all_images):
            repeat = int(repeats[i])*fps
            if repeat>1: all_images[i] = img.expand(repeat, *img.shape).flatten(0,1)
            else: all_images[i] = img

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
    RETURN_NAMES = ("latents","num_latents")
    RETURN_TYPES = ("LATENT","INT")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)

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
        num_latents = 0
        for i in range(len(samples)):
            repeat = int(repeats[i])
            samples[i]["batch_index"] = [i+offset for i in range(repeat)]
            offset += repeats[i]
            
            samples[i]["samples"] = LatentRepeatInterleavedNode.repeat(samples[i]["samples"],repeat)
            num_latents+=len(samples[i]["samples"])

            if "noise_mask" in samples[i]:
                samples[i]["noise_mask"] = LatentRepeatInterleavedNode.repeat(samples[i]["noise_mask"],repeat)

        return samples,num_latents

    def merge(self, latents, repeats, fps=1):
        repeats = np.array(repeats).flatten()
        fps = np.array(fps)[0]
        print(f"{repeats=} {fps=}")

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
                    if "noise_mask" in latent:
                        print(f"{latent['noise_mask'].shape=}")
                        new_latent["noise_mask"] = latent["noise_mask"]
                    new_latents.append(new_latent)

        if len(repeats)==1:
            new_latents *= repeats[0]
            repeats = np.ones(len(new_latents))
        elif len(repeats)<len(new_latents): # expand repeat
            pad_width = len(new_latents) - len(repeats)
            repeats = np.pad(repeats, (0, pad_width), mode='constant', constant_values=1)

        new_latents,num_latents = self.rebatch(new_latents,repeats.astype(np.int16)*fps)
        gc_collect()
        print(f"{len(latents)=} {num_latents=}")

        return (new_latents,num_latents)
    
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

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("vhs_audio",)

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
        return {"ui": {"preview": [{"filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId}]}, "result": (lambda: audio_to_bytes(*merged_audio),)}
    
    @classmethod
    def IS_CHANGED(cls, audio1, audio2, sr="None", merge_type="median", normalize=False, audio3_opt=None, audio4_opt=None):
        audios = [audio() for audio in [audio1, audio2, audio3_opt, audio4_opt] if audio is not None]
        return get_hash(sr, merge_type, normalize, *audios)

class SimpleMathNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "n1": ("INT,FLOAT", { "default": None, "step": 0.1 }),
                "n2": ("INT,FLOAT", { "default": None, "step": 0.1 }),
                "round_up": ("BOOLEAN", {"default": False})
            },
            "required": {
                "operation": (["CONVERT","ADD","SUBTRACT","MULTIPLY","DIVIDE","MODULUS","MIN","MAX"], { "default": "CONVERT" }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    FUNCTION = "do_math"
    CATEGORY = CATEGORY

    def do_math(self, operation, n1 = None, n2 = None, round_up=False):
        a, b = np.array(n1).flatten(), np.array(n2).flatten()
        if operation=="ADD": number=a+b
        elif operation=="SUBTRACT": number=a-b
        elif operation=="MULTIPLY": number=a*b
        elif operation=="DIVIDE":
            assert not any(b==0), f"cannot divide by 0 ({b=})!"
            number=a/b
        elif operation=="MODULUS": number=a%b
        elif operation=="MIN": number=np.array(list(map(min,zip(a,b))))
        elif operation=="MAX": number=np.array(list(map(max,zip(a,b))))
        else: number=a if n1 is not None else b

        print(f"{a=} \n{operation=} \n{b=} \n{number=}")

        num_to_int = math.ceil if round_up else math.floor
        if len(number)>1: # handles list inputs
            return (list(map(num_to_int,number)), list(map(float,number)),)
        else: return (num_to_int(number[0]), float(number[0]), )

# FROM: https://github.com/theUpsider/ComfyUI-Logic/blob/master/nodes.py#L11
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
    
class SliceNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "start": ("INT", {"default": 0, "min": 0}),
                "end": ("INT", {"default": -1}),
            },
            "required": {
                "array": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("array",)
    FUNCTION = "slice"
    CATEGORY = CATEGORY

    def slice(self, array, start=0, end=-1):
        if end==-1: end=len(array)
        return (array[start:end],)

class ZipImagesNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images1": ("IMAGE",),
                "images2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "dozip"
    CATEGORY = CATEGORY

    def dozip(self, images1, images2):
        return (list(map(torch.stack,zip(images1,images2))),)
    
class Any2ListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "to"
    CATEGORY = CATEGORY

    def to(self, any):
        return (list(any),)
    
class List2AnyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    INPUT_IS_LIST = (True, )
    FUNCTION = "to"
    CATEGORY = CATEGORY

    def to(self, any):
        return (any,)
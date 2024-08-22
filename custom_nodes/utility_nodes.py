import itertools
import math
import os
import torch
from .utils import AlwaysEqualProxy, MultipleTypeProxy
import folder_paths
from ..lib.utils import gc_collect
import numpy as np

CATEGORY = "🌺RVC-Studio/utils"
temp_path = folder_paths.get_temp_directory()

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

class SimpleMathNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "n1": (MultipleTypeProxy("INT,FLOAT"), { "default": None, "step": 0.1 }),
                "n2": (MultipleTypeProxy("INT,FLOAT"), { "default": None, "step": 0.1 }),
                "round_up": ("BOOLEAN", {"default": False})
            },
            "required": {
                "operation": (["CONVERT","ADD","SUBTRACT","MULTIPLY","DIVIDE","MODULUS","MIN","MAX"], { "default": "CONVERT" }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
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
            return (list(map(num_to_int,number)), list(map(float,number)), list(map(str,number)))
        else: return (num_to_int(number[0]), float(number[0]), str(number[0]))
    
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
    
class SortImagesNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "indices": ("INT", {"forceInput": True}),
                "reverse": ("BOOLEAN", {"default": False}),
                "sort_by": (["sum","mean","median","min","max"],{"default": "sum"})
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "indices")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, images, indices=None, reverse=False, sort_by="sum"):
        if sort_by=="mean": func=np.mean
        elif sort_by=="median": func=np.median
        elif sort_by=="min": func=np.amin
        elif sort_by=="max": func=np.amax
        else: func=np.sum
        values = list(map(lambda x: func(x.numpy()),images))
        if indices is None:
            indices = np.argsort(values)
        if reverse: indices=indices[::-1]
        indices = list(indices)
        return (images[indices],indices)
    
NODE_CLASS_MAPPINGS = {
    "MergeImageBatches": MergeImageBatches,
    "MergeLatentBatches": MergeLatentBatches,
    "ImageRepeatInterleavedNode": ImageRepeatInterleavedNode,
    "LatentRepeatInterleavedNode": LatentRepeatInterleavedNode,
    "SimpleMathNode": SimpleMathNode,
    "SliceNode": SliceNode,
    "ZipNode": ZipImagesNode,
    "Any2ListNode": Any2ListNode,
    "List2AnyNode": List2AnyNode,
    "SortImagesNode": SortImagesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeImageBatches": "🌺Merge Image Batches",
    "MergeLatentBatches": "🌺Merge Latent Batches",
    "ImageRepeatInterleavedNode": "🌺Image Repeat Interleaved",
    "LatentRepeatInterleavedNode": "🌺Latent Repeat Interleaved",
    "SimpleMathNode": "🌺Simple Math Operations",
    "SliceNode": "🌺Slice Array",
    "ZipNode": "🌺Zip Images",
    "Any2ListNode": "🌺Any to List",
    "List2AnyNode": "🌺List to Any",
    "SortImagesNode": "🌺Sort Images",
}
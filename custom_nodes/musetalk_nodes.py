
import json
from multiprocessing.pool import ThreadPool
import os
import cv2
import copy
from tqdm import tqdm

from ..lib.audio import get_audio, remix_audio
from ..lib import BASE_CACHE_DIR
from ..lib.utils import gc_collect, get_hash, get_optimal_torch_device
from .utils import MultipleTypeProxy, model_downloader
from ..lib.musetalk.models.unet import PositionalEncoding, UNet
from ..lib.musetalk.models.vae import VAE
from ..lib.musetalk.whisper.audio2feature import Audio2Feature
import torch
import numpy as np
import folder_paths
from ..lib.musetalk.utils.face_parsing import FaceParsing
from ..lib.musetalk.utils.blending import get_image
from ..lib.musetalk.utils.utils import datagen
import torchvision.transforms as TT
import torchvision
from PIL import Image

CATEGORY = "🌺RVC-Studio/musetalk"
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()
temp_path = folder_paths.get_temp_directory()
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def tensor_to_opencv(tensor_image: torch.Tensor):
    numpy_image = tensor_image.detach().numpy()
    if numpy_image.max()<=1: numpy_image *= 255
    opencv_image = cv2.cvtColor(numpy_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    opencv_image
    return opencv_image

def process_frame(i, res_frame, coord_list_cycle, frame_list_cycle, frames_dir, results_dir, fp_model):
    bbox = coord_list_cycle[i % (len(coord_list_cycle))]
    bbox = list(map(int, bbox))
    ori_frame = cv2.imread(os.path.join(frames_dir,frame_list_cycle[i % (len(frame_list_cycle))]))
    x1, y1, x2, y2 = bbox
    
    try:
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
        combined_frame = get_image(fp_model, ori_frame, res_frame, bbox)
        cv2.imwrite(os.path.join(results_dir, f"{str(i).zfill(8)}.png"), combined_frame)
    except Exception as error:
        print(f"{error=}")

def get_imagefiles(directory_path):
    assert os.path.isdir(directory_path), f"{directory_path} is not a directory!"

    files = sorted(fname for fname in os.listdir(directory_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')))    

    assert len(files)>0, f"{directory_path} is empty!"
    return files
        
def load_images_from_directory(directory_path):
    image_list = []
    img_size = None
    img_to_tensor = TT.ToTensor()

    files = get_imagefiles(directory_path)
    
    # Iterate over all files in the directory
    for filename in tqdm(files):
        # Load the image
        img_path = os.path.join(directory_path, filename)
        image = Image.open(img_path).convert('RGB')
        
        # Infer image size from the first image
        if img_size is None:
            img_size = image.size  # (width, height)
        
        # Resize image if it's not the same size as the first image
        if image.size != img_size:
            image = image.resize(img_size)
        
        # Convert image to tensor (HWC)
        tensor_image = img_to_tensor(image).permute(1, 2, 0)
        
        image_list.append(tensor_image)
    
    # Stack all images to form a batch (B, H, W, C)
    image_batch = torch.stack(image_list, dim=0)
    
    return image_batch

class MuseAudioFeatureExtractionNode:

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "audio": (MultipleTypeProxy("AUDIO,VHS_AUDIO"),),
            },
            "optional": {
                "fps": (MultipleTypeProxy("FLOAT,INT"),{"default": 1.}),
            }
        }

    CATEGORY = CATEGORY

    RETURN_TYPES = ("WHISPER_CHUNKS",)
    RETURN_NAMES = ("whisper_chunks",)
    FUNCTION = "process"

    def process(self, audio, fps):
        print("############################################## extract audio feature ##############################################")
        model_path = model_downloader("musetalk/whisper/tiny.pt")
        audio_processor = Audio2Feature(model_path=model_path)
        input_audio = remix_audio(get_audio(audio),target_sr=16000,norm=True)
        whisper_feature = audio_processor.audio2feat(np.array(input_audio[0],dtype=np.float32))
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        del audio_processor, whisper_feature, input_audio
        gc_collect()

        return (whisper_chunks,)

class MuseImageFeatureExtractionNode:

    device = get_optimal_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "images": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR", )
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "use_cache": ("BOOLEAN",{"default": True})
            }
        }

    CATEGORY = CATEGORY

    RETURN_TYPES = ("COORDS","STRING")
    RETURN_NAMES = ("coords","frames_dir")
    FUNCTION = "process"

    @staticmethod
    def mask_to_bbox(mask):
        # Ensure the mask is a binary tensor
        if mask is None: return (0.0,0.0,0.0,0.0)
        elif mask.dtype != torch.bool: mask = mask > 0
        
        # Find the bounding box coordinates using torchvision's masks_to_boxes
        boxes = torchvision.ops.masks_to_boxes(mask.unsqueeze(0))
        
        # Convert the boxes tensor to a list of coordinates
        x_min, y_min, x_max, y_max = boxes[0].tolist()
        
        return x_min, y_min, x_max, y_max

    def process(self, images, bbox_detector, threshold=.5, dilation=0, use_cache=True):
        print("############################################## preprocess input image  ##############################################")
        cache_name = get_hash(images,bbox_detector,len(images),threshold,dilation)
        frames_dir = os.path.join(temp_path,f"frames-{cache_name}")
        crop_coord_save_path = os.path.join(BASE_CACHE_DIR,"musetalk",f"coords-{cache_name}.json")

        if os.path.isfile(crop_coord_save_path) and os.path.isdir(frames_dir) and use_cache:
            print("using extracted coordinates")
            with open(crop_coord_save_path,'r') as f:
                data = json.load(f)
            coord_list = data["coord_list"]
        else:
            os.makedirs(frames_dir,exist_ok=True)
            print("extracting landmarks...time consuming")
            coord_list = []
            has_face = False
            for i,image in enumerate(tqdm(images.cpu())):
                mask = bbox_detector.detect_combined(image.unsqueeze(0), threshold, dilation)
                if mask is not None: has_face=True
                coords = self.mask_to_bbox(mask)
                coord_list.append(coords)
                cv2.imwrite(os.path.join(frames_dir, f"{str(i).zfill(8)}.png"), tensor_to_opencv(image))
            assert has_face, "No face detected!"

            if use_cache:
                os.makedirs(os.path.dirname(crop_coord_save_path),exist_ok=True)
                with open(crop_coord_save_path, 'w') as f:
                    json.dump(dict(coord_list=coord_list), f)
                
        gc_collect()

        return (coord_list, frames_dir)
    
class MuseTalkNode:

    device = get_optimal_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "coord_list": ("COORDS",),
                "frames_dir": ("STRING",{"default": ""}),
                "whisper_chunks": ("WHISPER_CHUNKS",)
            },
             "optional": {
                "batch_size": ("INT",{"default": 1, "min": 1}),
             }
        }

    CATEGORY = CATEGORY

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("images","results_dir")
    FUNCTION = "process"

    def process(self, coord_list, frames_dir, whisper_chunks, batch_size=1):
        frame_list = get_imagefiles(frames_dir)
        results_dir = os.path.join(temp_path,f"results-{get_hash(coord_list, whisper_chunks, *frame_list)}")

        if not os.path.isdir(results_dir) or len(os.listdir(results_dir))==0:
            os.makedirs(results_dir,exist_ok=True)

            print("############################################## process latents ##############################################")
            vae_model = model_downloader("musetalk/sd-vae-ft-mse/diffusion_pytorch_model.safetensors")
            _ = model_downloader("musetalk/sd-vae-ft-mse/config.json")
            vae = VAE(model_path = os.path.dirname(vae_model), use_float16=True)
            
            input_latent_list = []
            for bbox, frame in tqdm(zip(coord_list, frame_list)):
                bbox = list(map(int,bbox))
                frame = cv2.imread(os.path.join(frames_dir,frame))
                if sum(bbox) == 0:
                    print(f"No face detected: {bbox}")
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
        
            # to smooth the first and the last frame
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            print(f"{len(frame_list_cycle)=} {len(coord_list_cycle)=} {len(input_latent_list_cycle)=} {len(whisper_chunks)=}")

            print("############################################## inference batch by batch ##############################################")
            unet_config = model_downloader("musetalk/musetalk.json")
            unet_model = model_downloader("musetalk/pytorch_model.bin")
            unet = UNet(unet_config=unet_config,model_path=unet_model,use_float16=True)
            pe = PositionalEncoding(d_model=384)
            video_num = len(whisper_chunks)
            dataset = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
            res_frame_list = []
            timestep = torch.tensor([0], device=unet.device)
            for (whisper_batch,latent_batch) in tqdm(dataset,total=int(np.ceil(float(video_num)/batch_size))):
                tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
                audio_feature_batch = torch.stack(tensor_list) # torch, B, 5*N,384
                audio_feature_batch = pe(audio_feature_batch)
                
                pred_latents = unet.model(latent_batch.to(unet.device).half(), timestep, encoder_hidden_states=audio_feature_batch.to(unet.device).half()).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            del pe, unet.model, vae, unet
            gc_collect()

            print("############################################## pad to full image ##############################################")
            resnet_path = model_downloader('musetalk/face-parse-bisent/resnet18-5c106cde.pth')
            face_model_pth = model_downloader('musetalk/face-parse-bisent/79999_iter.pth')
            fp_model = FaceParsing(resnet_path,face_model_pth)

            with ThreadPool(batch_size) as pool:
                for i, res_frame in enumerate(tqdm(res_frame_list)):
                    pool.apply(process_frame, args=(i, res_frame, coord_list_cycle, frame_list_cycle, frames_dir, results_dir, fp_model))
                
            del fp_model, frame_list_cycle, res_frame_list, coord_list_cycle, input_latent_list
            gc_collect()

        images = load_images_from_directory(results_dir)

        print(f"{images.shape=} {images.max()=} {images.dtype=}")

        return (images,results_dir)
    
NODE_CLASS_MAPPINGS = {
    "MuseTalkNode": MuseTalkNode,
    "MuseAudioFeatureExtractionNode": MuseAudioFeatureExtractionNode,
    "MuseImageFeatureExtractionNode": MuseImageFeatureExtractionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseTalkNode": "🌺MuseTalk Processor",
    "MuseAudioFeatureExtractionNode": "🌺MuseTalk Feature Extractor",
    "MuseImageFeatureExtractionNode": "🌺MuseTalk Image Preprocessing"
}
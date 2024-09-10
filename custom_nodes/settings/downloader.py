import platform
import re
import shutil
import subprocess
from typing import IO, List, Tuple
import unicodedata
import requests
import os
from zipfile import ZipFile

from ...lib import BASE_CACHE_DIR, BASE_MODELS_DIR

RVC_DOWNLOAD_LINK = 'https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/'
MDX_MODELS = ["MDXNET/UVR-MDX-NET-vocal_FT.onnx"]
KARAFAN_MODELS = ["karafan/MDX23C-8KFFT-InstVoc_HQ.ckpt"]
VR_MODELS = [
    "UVR/UVR-DeEcho-DeReverb.pth",
    "UVR/HP5-vocals+instrumentals.pth",
    "UVR/5_HP-Karaoke-UVR.pth",
    "UVR/6_HP-Karaoke-UVR.pth",
    "UVR/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "UVR/UVR-BVE-4B_SN-44100-1.pth",
    "UVR/UVR-DeNoise.pth"
]
RVC_MODELS = [
    "RVC/Claire.pth",
    "RVC/Sayano.pth",
    "RVC/Mae_v2.pth",
    "RVC/Fuji.pth",
    "RVC/Monika.pth"
]
RVC_INDEX = [
    "RVC/.index/added_IVF1063_Flat_nprobe_1_Sayano_v2.index",
    "RVC/.index/added_IVF985_Flat_nprobe_1_Fuji_v2.index",
    "RVC/.index/Monika_v2_40k.index",
    "RVC/.index/Sayano_v2_40k.index"
]
BASE_MODELS = ["content-vec-best.safetensors", "rmvpe.pt"]
VITS_MODELS = ["VITS/pretrained_ljs.pth"]
PRETRAINED_MODELS_G = [
    "pretrained_v2/G48k.pth",
    "pretrained_v2/G32k.pth",
    "pretrained_v2/G40k.pth",
    "pretrained_v2/f0G48k.pth",
    "pretrained_v2/f0G40k.pth",
    "pretrained_v2/f0G32k.pth",
    "pretrained_v2/f0_RIN_E3_40k_G.pth",
    "pretrained_v2/f0Ov2Super32kG.pth",
    "pretrained_v2/f0Ov2Super40kG.pth",
]
PRETRAINED_MODELS_D = [
    "pretrained_v2/D48k.pth",
    "pretrained_v2/D32k.pth",
    "pretrained_v2/D40k.pth",
    "pretrained_v2/f0D48k.pth",
    "pretrained_v2/f0D40k.pth",
    "pretrained_v2/f0D32k.pth",
    "pretrained_v2/f0_RIN_E3_40k_D.pth",
    "pretrained_v2/f0Ov2Super32kD.pth",
    "pretrained_v2/f0Ov2Super40kD.pth",
]
LLM_MODELS = [
    "https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF/resolve/main/airoboros-l2-7b-2.1.Q4_K_M.gguf",
    "https://huggingface.co/TheBloke/Pygmalion-2-7B-GGUF/resolve/main/pygmalion-2-7b.Q4_K_M.gguf",
    "https://huggingface.co/TheBloke/Zarablend-MX-L2-7B-GGUF/resolve/main/zarablend-mx-l2-7b.Q4_K_M.gguf",
    "https://huggingface.co/TheBloke/MythoMax-L2-Kimiko-v2-13B-GGUF/resolve/main/mythomax-l2-kimiko-v2-13b.Q4_K_M.gguf"
]
STT_MODELS = [
    "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
]

def download_file(params: Tuple[str, str]):
    model_path, download_link = params
    if os.path.isfile(model_path): raise FileExistsError(f"{model_path} already exists!")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with requests.get(download_link,stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

def download_link_generator(download_link: str,model_list: List[str]):
    for model in model_list:
        model_path = os.path.join(BASE_MODELS_DIR,model)
        yield (model_path, f"{download_link}{model}")

def save_file(params: Tuple[str, any]):
    (data_path, datum) = params
    if "zip" in os.path.splitext(data_path)[-1]: return save_zipped_files(params) # unzip
    else:
        try:
            with open(data_path,"wb") as f:
                f.write(datum)
            return f"Successfully saved file to: {data_path}"
        except Exception as e:
            return f"Failed to save file: {e}"

def save_file_generator(save_dir: str, data: List[IO]):
    for datum in data:
        data_path = os.path.join(save_dir,datum.name)
        yield (data_path, datum.read())

def extract_zip_without_structure(zip_path, extract_to, cleanup=False):
    os.makedirs(extract_to,exist_ok=True)
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zipped_files = zip_ref.namelist()
            extracted_files = os.listdir(extract_to)
            if len(zipped_files)>len(extracted_files):
                for member in zipped_files:
                    # Get the file name only (discard the directory structure)
                    filename = os.path.basename(member)
                    if filename:  # Check if it's not an empty string
                        # Create the full path for the extracted file
                        file_path = os.path.join(extract_to, filename)
                        # Extract the file
                        with zip_ref.open(member) as source, open(file_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
        if cleanup: os.remove(zip_path) # cleanup
        print(f"Successfully extracted files to: {extract_to}")
    except Exception as error:
        print(f"Failed to extract files: {error}")
    finally: return os.listdir(extract_to)
                    
def save_zipped_files(params: Tuple[str, any]):
    (data_path, datum) = params

    try:
        print(f"saving zip file: {data_path}")
        temp_dir = os.path.join(BASE_CACHE_DIR,"zips")
        os.makedirs(temp_dir,exist_ok=True)
        name = os.path.basename(data_path)
        zip_path = os.path.join(temp_dir,name)

        with open(zip_path,"wb") as f:
            f.write(datum)

        print(f"extracting zip file: {zip_path}")
        files = extract_zip_without_structure(zip_path,os.path.dirname(data_path),cleanup=True)
        print(f"finished extracting {len(files)} files")
        
        return True
    except Exception as e:
        print(f"Failed to save files: {e}")
        return False
    
def slugify_filepath(filepath):
    # Split the path into directory and filename
    directory, filename = os.path.split(filepath)
    # Normalize the filename
    filename = unicodedata.normalize('NFKD', filename)
    # Encode the filename as ASCII and ignore errors
    filename = filename.encode('ascii', 'ignore').decode()
    # Convert the filename to lowercase
    filename = filename.lower()
    # Replace spaces and other unwanted characters with dashes
    filename = re.sub(r'[^a-z0-9.-]+', '-', filename)
    # Join the directory and the slugified filename
    return os.path.join(directory, filename)

def download_ffmpeg():
    if platform.system() == "Windows":
        link = f"{RVC_DOWNLOAD_LINK}ffmpeg.exe"
        ffmpeg_path = os.path.join(os.getcwd(),"ffmpeg.exe")
        if os.path.isfile(ffmpeg_path): return True
        return download_file((ffmpeg_path, link))
    elif platform.system() == "Linux":
        subprocess.check_call("apt update && apt install -y -qq ffmpeg espeak")
        return True
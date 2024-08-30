import json
import multiprocessing
import os
import shutil

import numpy as np
import torch

from . import BASE_DIR

from ..lib.train.utils import HParams

from ..training_cli import train_model

from ..preprocessing_utils import extract_features_trainset, preprocess_trainset

from ..config import config

from ..lib.model_utils import load_hubert

from .utils import MultipleTypeProxy, increment_filename_no_overwrite, model_downloader
from .settings import PITCH_EXTRACTION_OPTIONS
from .settings.downloader import PRETRAINED_MODELS, RVC_DOWNLOAD_LINK, RVC_INDEX, RVC_MODELS, download_file, extract_zip_without_structure

from ..lib.audio import SUPPORTED_AUDIO, audio_to_bytes, load_input_audio, save_input_audio, get_audio, SR_MAP

from ..vc_infer_pipeline import get_vc, vc_single
import folder_paths
from ..lib.utils import get_filenames, get_hash, get_optimal_threads, get_optimal_torch_device
from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR

input_path = folder_paths.get_input_directory()
temp_path = folder_paths.get_temp_directory()
cache_dir = os.path.join(BASE_CACHE_DIR,"rvc")
output_path = folder_paths.get_output_directory()
node_path = os.path.join(BASE_MODELS_DIR,"custom_nodes/ComfyUI-UVR5")
weights_path = os.path.join(BASE_MODELS_DIR, "uvr5")
dataset_path = os.path.join(input_path,"datasets")
device = get_optimal_torch_device()
CATEGORY = "🌺RVC-Studio/rvc"
MUTE_DIR = os.path.join(BASE_DIR,"dataset")

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
                }),
                "resample_sr": ([0,16000,32000,40000,44100,48000],{"default": 0}),
                "rms_mix_rate": ("FLOAT",{
                    "default": 0.25, 
                    "min": 0., #Minimum value
                    "max": 1., #Maximum value
                    "step": .01, #Slider's step
                }),
                "protect": ("FLOAT",{
                    "default": 0.25, 
                    "min": 0., #Minimum value
                    "max": .5, #Maximum value
                    "step": .01, #Slider's step
                }),
                "crepe_hop_length": ("INT",{"default": 160, "min": 16, "max": 512, "step": 16})
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
    
class LoadRVCModelNode:

    @classmethod
    def INPUT_TYPES(cls):
        model_list = RVC_MODELS + get_filenames(root=BASE_MODELS_DIR,folder="RVC",exts=["pth"],format_func=lambda x: f"RVC/{os.path.basename(x)}")
        model_list = list(set(model_list)) # dedupe
        index_list = [""] + RVC_INDEX + get_filenames(root=os.path.join(BASE_MODELS_DIR,"RVC"),folder=".index",exts=["index"],format_func=lambda x: f"RVC/.index/{os.path.basename(x)}")
        index_list = list(set(index_list)) # dedupe

        return {
            'required': {
                'model': (model_list,{"default": model_list[0]}),
            },
            "optional": {
                "index": (index_list,{"default": ""}),
            }
        }

    RETURN_TYPES = ('RVC_MODEL', 'STRING')
    RETURN_NAMES = ('model', 'model_name')

    CATEGORY = CATEGORY

    FUNCTION = 'load_model'

    def load_model(self, model, index=""):
        model_path = file_index = None
        try:
            filename = os.path.basename(model)
            subfolder = os.path.dirname(model)
            model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
            
            if not os.path.isfile(model_path):
                download_link = f"{RVC_DOWNLOAD_LINK}{model}"
                if download_file((model_path, download_link)): print(f"successfully downloaded: {model_path}")

            if index:
                file_index = os.path.join(BASE_MODELS_DIR,subfolder,".index",os.path.basename(index))
                if not os.path.isfile(file_index):
                    download_link = f"{RVC_DOWNLOAD_LINK}{index}"
                    if download_file((file_index, download_link)): print(f"successfully downloaded: {file_index}")
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {e}")
            raise e
        finally: return (lambda:get_vc(model_path, file_index),filename.split(".")[0])
    
class RVCNode:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):


        return {
            "required": {
                "audio": (MultipleTypeProxy('AUDIO,VHS_AUDIO'),),
                "model": ("RVC_MODEL",),
                "hubert_model": ("HUBERT_MODEL",),
                "pitch_extraction_params": ("PITCH_EXTRACTION",),
                "f0_up_key": ("INT",{
                    "default": 0, 
                    "min": -14, #Minimum value
                    "max": 14, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider"
                }),
            },
            "optional": {
                "format":(SUPPORTED_AUDIO,{"default": "flac"}),
                "use_cache": ("BOOLEAN",{"default": True})
            }
        }
    
    OUTPUT_NODE = True

    RETURN_TYPES = ("VHS_AUDIO",)

    FUNCTION = "convert"

    CATEGORY = CATEGORY

    def convert(self, audio, model, hubert_model, pitch_extraction_params, f0_up_key, format="flac", use_cache=True):
        
        input_audio = get_audio(audio)
        voice_model = model()
        feature_model = hubert_model()
        widgetId = get_hash(feature_model, f0_up_key, audio_to_bytes(*input_audio), *voice_model.items(), *pitch_extraction_params.items())
        cache_name = os.path.join(BASE_CACHE_DIR,"rvc",f"{widgetId}.{format}")

        if use_cache and os.path.isfile(cache_name): output_audio = load_input_audio(cache_name)
        else:
            output_audio = vc_single(hubert_model=feature_model,input_audio=input_audio,f0_up_key=f0_up_key,**voice_model,**pitch_extraction_params)
            
            if use_cache:
                print(save_input_audio(cache_name, output_audio))
                if os.path.isfile(cache_name): output_audio = load_input_audio(cache_name)
        
        tempdir = os.path.join(temp_path,"preview")
        os.makedirs(tempdir, exist_ok=True)
        audio_name = os.path.basename(cache_name)
        preview_file = os.path.join(tempdir,audio_name)
        if not os.path.isfile(preview_file): shutil.copyfile(cache_name,preview_file)
        return {"ui": {"preview": [{"filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId}]}, "result": (lambda:audio_to_bytes(*output_audio),)}
    
class RVCProcessDatasetNode:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):

        os.makedirs(dataset_path, exist_ok=True)
        DATASETS = [""] + [dname for dname in os.listdir(dataset_path) if dname.endswith("zip")]

        return {
            "required": {
                "model_name": ("STRING",{"default": ""}),
                "dataset": (DATASETS, {"default": ""}),
                "hubert_model": ("HUBERT_MODEL",),
            },
            "optional": {
                "pitch_extraction_params": ("PITCH_EXTRACTION",{"default": {}}),
                "sr": (["32k","40k","48k"], {"default": "40k"}),
                "n_threads": ("INT", {"default": get_optimal_threads(), "min": 1, "max": multiprocessing.cpu_count()}),
                "period": ("FLOAT", {"default": 3., "min": 1., "max": 10., "step": .1}),
                "overlap": ("FLOAT",{"default": .3, "min": .1, "max": 1., "step": .1}),
                "max_volume": ("FLOAT",{"default": .99, "min": .1, "max": 1., "step": .01}),
                "mute_ratio": ("FLOAT",{"default": .0, "min": .0, "max": .5, "step": .01}),
                "audio_processor": ("AUDIO_PROCESSOR",)
            }
        }

    RETURN_TYPES = ("RVC_DATASET_PIPE",)
    RETURN_NAMES = ("rvc_dataset_pipe",)

    FUNCTION = "process"

    CATEGORY = CATEGORY

    def process(self, model_name: str, dataset: str, hubert_model, pitch_extraction_params={}, sr="40k", n_threads=1, period=3., overlap=.3, max_volume=1., mute_ratio=.0, audio_processor=None):
        
        assert model_name, "Please provide a model name!"
        assert dataset, "Please upload a dataset!"
        
        f0_method = pitch_extraction_params.get("f0_method", "")
        cached_params = [model_name, dataset, period, overlap, max_volume, mute_ratio, sr, f0_method, audio_processor]
        crepe_hop_length = pitch_extraction_params.get("crepe_hop_length",160)

        if "crepe" in f0_method: cached_params.append(crepe_hop_length)
        
        cache_name = get_hash(*cached_params)
        dataset_dir = os.path.join(output_path,"dataset",cache_name)
        os.makedirs(dataset_dir,exist_ok=True)

        filelist_path = os.path.join(dataset_dir, "filelist.txt")
        
        if not os.path.isfile(filelist_path):
            input_dir = os.path.join(dataset_path,dataset.split(".")[0])

            if dataset.endswith("zip"):
                files = extract_zip_without_structure(os.path.join(dataset_path,dataset),input_dir)
                assert len(files), "Failed to extract zip file..."
            
            assert preprocess_trainset(input_dir,SR_MAP[sr],n_threads,dataset_dir,audio_processor,period,overlap,max_volume), "Failed to preprocess audio..."
            
            assert extract_features_trainset(hubert_model(), dataset_dir,n_p=n_threads,f0method=f0_method,device=device,if_f0=bool(f0_method),version="v2",crepe_hop_length=crepe_hop_length), "Failed to extract features..."

            gt_wavs_dir = os.path.join(dataset_dir,"0_gt_wavs")
            feature_dir = os.path.join(dataset_dir,"3_feature768")
            os.makedirs(gt_wavs_dir, exist_ok=True)
            os.makedirs(feature_dir, exist_ok=True)
            
            # add training data 
            if f0_method:
                f0_dir =  os.path.join(dataset_dir,"2a_f0")
                f0nsf_dir = os.path.join(dataset_dir,"2b-f0nsf")
                names = (
                    set([os.path.splitext(name)[0] for name in os.listdir(feature_dir)])
                    & set([os.path.splitext(name)[0] for name in os.listdir(f0_dir)])
                    & set([os.path.splitext(name)[0] for name in os.listdir(f0nsf_dir)])
                )
            else:
                names = set(
                    [os.path.splitext(name)[0] for name in os.listdir(feature_dir)]
                )
            opt = []
            missing_data = []
            for name in names:
                name_parts = name.split(",")
                gt_name = name if len(name_parts) == 1 else name_parts[-1]
                gt_file = os.path.join(gt_wavs_dir,gt_name)
                if not os.path.isfile(gt_file):
                    print(f"{gt_name} not found!")
                    missing_data.append(gt_name)
                    continue #skip data

                if f0_method:
                    data = "|".join([
                        gt_file,
                        os.path.join(feature_dir,f"{name}.npy"),
                        os.path.join(f0_dir,f"{name}.npy"),
                        os.path.join(f0nsf_dir,f"{name}.npy"),
                        str(0)
                    ])
                else:
                    data = "|".join([
                        gt_file,
                        os.path.join(feature_dir,f"{name}.npy"),
                        str(0)
                    ])
                opt.append(data)

            assert len(missing_data)==0, f"missing ground truth data: {len(opt)=}, {len(missing_data)=}"

            # add mute data 
            fea_dim = 768
            num_mute = max(2,int(len(opt)*mute_ratio)) # use 1% mute file or 2 copies (like original repo)
            for _ in range(num_mute):
                if f0_method:
                    data = "|".join([
                        os.path.join(MUTE_DIR,"mute","0_gt_wavs",f"mute{sr}.wav"),
                        os.path.join(MUTE_DIR,"mute",f"3_feature{fea_dim}","mute.npy"),
                        os.path.join(MUTE_DIR,"mute","2a_f0","mute.wav.npy"),
                        os.path.join(MUTE_DIR,"mute","2b-f0nsf","mute.wav.npy"),
                        str(0)
                    ])
                else:
                    data = "|".join([
                        os.path.join(MUTE_DIR,"mute","0_gt_wavs",f"mute{sr}.wav"),
                        os.path.join(MUTE_DIR,"mute",f"3_feature{fea_dim}","mute.npy"),
                        str(0)
                    ])
                opt.append(data)


            np.random.shuffle(opt)
            with open(filelist_path, "w") as f:
                f.write("\n".join(opt))
            print("write filelist done")
        return (dict(
            sample_rate=sr,
            dataset_dir=dataset_dir,
            name=model_name,
            training_files=filelist_path,
            if_f0=bool(f0_method),
            pitch_extraction_params=pitch_extraction_params,
            hubert_model=hubert_model
            ),)


class RVCTrainModelNode:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        DEVICES = [str(i) for i in range(torch.cuda.device_count())]
        PRETRAINED_G = [model for model in PRETRAINED_MODELS if "G" in model] + get_filenames(root=BASE_MODELS_DIR,folder="pretrained_v2",name_filters=["G"],format_func=lambda x: f"pretrained_v2/{os.path.basename(x)}")
        PRETRAINED_G = list(set(PRETRAINED_G)) # dedupe
        PRETRAINED_D = [model for model in PRETRAINED_MODELS if "D" in model] + get_filenames(root=BASE_MODELS_DIR,folder="pretrained_v2",name_filters=["D"],format_func=lambda x: f"pretrained_v2/{os.path.basename(x)}")
        PRETRAINED_D = list(set(PRETRAINED_D)) # dedupe
        
        return {
            "required": {
                "rvc_dataset_pipe": ("RVC_DATASET_PIPE",),
            },
            "optional": dict(
                gpu=(DEVICES, {"default": DEVICES[0]}),
                batch_size=("INT",dict(default=4,min=1,max=64,step=1)),
                total_epoch=("INT",dict(default=100,min=10,max=1000,step=10)),
                save_every_epoch=("INT",dict(default=0,min=0,max=100)),
                pretrained_G=(PRETRAINED_G,{"default": PRETRAINED_G[0]}),
                pretrained_D=(PRETRAINED_D,{"default": PRETRAINED_D[0]}),
                if_save_latest=("BOOLEAN",{"default":True}),
                if_cache_gpu=("BOOLEAN",{"default":True}),
                if_save_every_weights=("BOOLEAN",{"default":False}),
                train_index=("BOOLEAN",{"default": True}),
                retrain=("BOOLEAN",{"default": False}),
                save_best_model=("BOOLEAN",{"default": True}),
                log_every_epoch=("FLOAT",dict(default=1.,min=0.,max=2.,step=.5)),
                gradient_lambda=("FLOAT",dict(default=0.,min=0.,max=100.,step=.1)),
                timbre_lambda=("FLOAT",dict(default=0.,min=0.,max=100.,step=.1)),
                num_workers=("INT",dict(default=1,min=1,max=16))
            )
        }

    RETURN_TYPES = ('RVC_MODEL', 'STRING', "HUBERT_MODEL", "PITCH_EXTRACTION" )
    RETURN_NAMES = ('model', 'model_name', "hubert_model", "pitch_extraction_params" )

    OUTPUT_NODE = True

    FUNCTION = "train_model"

    CATEGORY = CATEGORY

    def train_model(self,
                    rvc_dataset_pipe,
                    gpu="0",
                    batch_size=4,
                    total_epoch=100,
                    save_every_epoch=0,
                    pretrained_G="",
                    pretrained_D="",
                    if_save_latest=True,
                    if_cache_gpu=True,
                    if_save_every_weights=False,
                    train_index=True,
                    retrain=False,
                    save_best_model=True,
                    log_every_epoch=1.,
                    gradient_lambda=0.,
                    timbre_lambda=0.,
                    num_workers=1):
        
        sample_rate = rvc_dataset_pipe["sample_rate"]
        name = rvc_dataset_pipe["name"]
        dataset_dir = rvc_dataset_pipe["dataset_dir"]
        cache_name = get_hash(batch_size,pretrained_G,pretrained_D,gradient_lambda,timbre_lambda)
        model_dir = os.path.join(output_path,"logs",cache_name)
        if_f0 = rvc_dataset_pipe["if_f0"]

        config_path = os.path.join(BASE_DIR,"configs",f"{sample_rate}{'' if sample_rate=='40k' else '_v2'}.json")
        with open(config_path,"r") as f:
            config = json.load(f)

        hparams = HParams(**config)
        hparams.experiment_dir = dataset_dir
        hparams.model_dir = model_dir
        hparams.save_every_epoch = save_every_epoch
        hparams.name = name
        hparams.total_epoch = total_epoch
        hparams.pretrainG = model_downloader(pretrained_G)
        hparams.pretrainD = model_downloader(pretrained_D)
        hparams.version = "v2"
        hparams.gpus = gpu
        hparams.train.batch_size = max(batch_size,1)
        hparams.sample_rate = sample_rate
        hparams.if_f0 = if_f0
        hparams.if_latest = if_save_latest
        hparams.save_every_weights = if_save_every_weights
        hparams.if_cache_data_in_gpu = if_cache_gpu
        hparams.data.training_files = rvc_dataset_pipe["training_files"]
        hparams.save_best_model = save_best_model
        hparams.log_every_epoch = log_every_epoch
        hparams.train.gradient_lambda = gradient_lambda
        hparams.train.num_workers = num_workers
        hparams.train.timbre_lambda = timbre_lambda

        file_index = self.train_index(dataset_dir, sample_rate, name) if train_index else None
        model_path = os.path.join(BASE_MODELS_DIR,"RVC",f"{name}_{sample_rate}.pth")
        if os.path.isfile(model_path) and retrain: model_path = increment_filename_no_overwrite(model_path)
        hparams.model_path = model_path

        if not os.path.isfile(model_path): train_model(hparams)
        assert os.path.isfile(model_path), f"Failed to train model {model_path}..."

        return (lambda: get_vc(model_path, file_index), name, rvc_dataset_pipe["hubert_model"], rvc_dataset_pipe["pitch_extraction_params"])

    def train_index(self, dataset_dir, sr, name):

        key = get_hash(dataset_dir, sr, name)
        index_file = os.path.join(BASE_MODELS_DIR,"RVC",".index",f"{name}_v2_{sr}_{key}.index")

        try:

            if not os.path.isfile(index_file):
                from sklearn.cluster import MiniBatchKMeans
                import faiss

                feature_dir = os.path.join(dataset_dir, "3_feature768")

                npys = []
                listdir_res = list(os.listdir(feature_dir))
                for fname in sorted(listdir_res):
                    phone = np.load(os.path.join(feature_dir, fname))
                    npys.append(phone)
                big_npy = np.concatenate(npys, 0)

                big_npy_idx = np.arange(big_npy.shape[0])
                np.random.shuffle(big_npy_idx)
                big_npy = big_npy[big_npy_idx]

                if big_npy.shape[0] > 2e5:
                    big_npy = (
                        MiniBatchKMeans(
                            n_clusters=10000,
                            verbose=True,
                            batch_size=256 * config.n_cpu,
                            compute_labels=False,
                            init="random",
                        )
                        .fit(big_npy)
                        .cluster_centers_
                    )

                n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
                print(f"{big_npy.shape=} {n_ivf=}")
                index = faiss.index_factory(768, "IVF%s,Flat" % n_ivf)
                print("training index")
                index_ivf = faiss.extract_index_ivf(index)  #
                index_ivf.nprobe = 1
                index.train(big_npy)
                print("adding index")
                batch_size_add = 8192
                for i in range(0, big_npy.shape[0], batch_size_add):
                    index.add(big_npy[i : i + batch_size_add])
                
                faiss.write_index(index,index_file)
                print(f"saved index file to {index_file}")
            return index_file
        except Exception as e:
            print(f"Failed to train index: {e}")
        return None
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadRVCModelNode": LoadRVCModelNode,
    "RVCNode": RVCNode,
    "LoadHubertModel": LoadHubertModel,
    "LoadPitchExtractionParams": LoadPitchExtractionParams,
    "RVCProcessDatasetNode": RVCProcessDatasetNode,
    "RVCTrainModelNode": RVCTrainModelNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRVCModelNode": "🌺Load RVC Model",
    "RVCNode": "🌺Voice Changer",
    "LoadHubertModel": "🌺Load Hubert Model",
    "LoadPitchExtractionParams": "🌺Load Pitch Extraction Params",
    "RVCProcessDatasetNode": "🌺Process RVC Dataset",
    "RVCTrainModelNode": "🌺Train RVC Model",
}
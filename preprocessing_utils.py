import sys, os, multiprocessing
from threading import Thread
import numpy as np, os, traceback
from .lib.slicer2 import Slicer
import traceback
from scipy.io import wavfile
from .pitch_extraction import FeatureExtractor
from .lib.audio import hz_to_mel, load_input_audio, remix_audio, AudioProcessor
from .lib.utils import gc_collect
from .config import config
import torch

class Preprocess:
    def __init__(self, sr, exp_dir, preprocessor: "AudioProcessor"=None, noparallel=True, period=3.0, overlap=.3, max_volume=.95):
        self.slicer = Slicer(
            sr=sr,
            threshold=-50,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500
        )
        self.sr = sr
        self.per = period
        self.overlap = overlap
        self.tail = self.per + self.overlap
        self.max_volume = max_volume
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir,"0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir,"1_16k_wavs")
        self.noparallel = noparallel
        self.preprocessor = preprocessor
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def println(self,strr):
        # mutex.acquire()
        print(strr)
        with open("%s/preprocess.log" % self.exp_dir, "a+") as f:
            f.write("%s\n" % strr)
            f.flush()
        # mutex.release()

    def norm_write(self, tmp_audio, idx0, idx1):
        if len(tmp_audio) > self.overlap*self.sr*2:
            wavfile.write(os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"),self.sr,tmp_audio.astype(np.float32))
            remixed_audio = remix_audio((tmp_audio, self.sr), target_sr=16000, max_volume=self.max_volume)
            wavfile.write(os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"),16000,remixed_audio[0].astype(np.float32))
        else: print(f"skipped short audio clip: {idx0}_{idx1}.wav ({len(tmp_audio)=})")

    def pipeline(self, path, idx0):
        try:
            input_audio = load_input_audio(path, self.sr)
            if self.preprocessor is not None: input_audio = self.preprocessor(input_audio)

            idx1 = 0
            for audio in self.slicer.slice(input_audio[0]):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            self.println("%s->Suc." % path)
        except:
            self.println("%s->%s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if self.noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            self.println("Fail. %s" % traceback.format_exc())

class FeatureInput(FeatureExtractor):
    def __init__(self, model, f0_method, exp_dir, samplerate=16000, hop_size=160, device="cpu", version="v2", if_f0=False):
        self.sr = samplerate
        self.hop = hop_size
        self.f0_method = f0_method
        self.exp_dir = exp_dir
        self.device = device
        self.version = version
        self.if_f0 = if_f0

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = hz_to_mel(self.f0_min)
        self.f0_mel_max = hz_to_mel(self.f0_max)

        self.model = model
        
        super().__init__(samplerate, config, onnx=False)

    def printt(self,strr):
        print(strr)
        with open("%s/extract_f0_feature.log" % self.exp_dir, "a+") as f:
            f.write("%s\n" % strr)
            f.flush()

    def compute_feats(self,x):
        feats = torch.from_numpy(x).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)

        inputs = {
            "source": feats.half().to(self.device) 
                if self.device not in ["mps", "cpu"]
                else feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 9 if self.version == "v1" else 12,  # layer 9
        }
        
        feats = self.model.extract_features(version=self.version,**inputs)

        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
            return feats
        else:
            return self.printt("==contains nan==")

    def compute_f0(self,x):
        return self.get_f0(x,0,self.f0_method,crepe_hop_length=self.hop)
    
    def go(self, paths):
        if len(paths) == 0:
            self.printt("no-f0-todo")
        else:
            self.printt("todo-f0-%s" % len(paths))
            # n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2, opt_path3) in enumerate(paths):
                try:
                    # if idx % n == 0:
                    #     self.printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                        and os.path.exists(opt_path3 + ".npy") == True
                    ):
                        continue
                    x,_ = load_input_audio(inp_path,self.sr)
                    if self.model:
                        feats = self.compute_feats(x)
                        if feats is not None:
                            np.save(
                                opt_path3,
                                feats,
                                allow_pickle=False,
                            )  # features
                            if self.if_f0: # uses pitch
                                coarse_pit, featur_pit = self.compute_f0(x)
                                np.save(
                                    opt_path2,
                                    featur_pit,
                                    allow_pickle=False,
                                )  # nsf
                                np.save(
                                    opt_path1,
                                    coarse_pit,
                                    allow_pickle=False,
                                )  # ori
                except:
                    self.printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))

def preprocess_trainset(inp_root, sr, n_p, exp_dir, preprocessor=None, period=3.0, overlap=.3, max_volume=1.):
    try:
        pp = Preprocess(sr, exp_dir, preprocessor=preprocessor, period=period, overlap=overlap, max_volume=max_volume)
        pp.println("start preprocess")
        pp.println(sys.argv)
        pp.pipeline_mp_inp_dir(inp_root, n_p)
        pp.println("end preprocess")
        del pp
        gc_collect()
        print("Successfully preprocessed data")
        return True
    except Exception as e:
        print(f"Failed to preprocess data: {e}")
        return False

def extract_features_trainset(hubert_model,exp_dir,n_p,f0method,device,version,if_f0,crepe_hop_length):
    try:
        featureInput = FeatureInput(f0_method=f0method,exp_dir=exp_dir,device=device,version=version,if_f0=if_f0,model=hubert_model,hop_size=crepe_hop_length)
        paths = []
        inp_root = os.path.join(exp_dir,"1_16k_wavs")
        opt_root1 = os.path.join(exp_dir,"2a_f0")
        opt_root2 = os.path.join(exp_dir,"2b-f0nsf")
        opt_root3 = os.path.join(exp_dir,"3_feature256" if version == "v1" else "3_feature768")

        os.makedirs(opt_root1, exist_ok=True)
        os.makedirs(opt_root2, exist_ok=True)
        os.makedirs(opt_root3, exist_ok=True)

        for name in sorted(list(os.listdir(inp_root))):
            inp_path = os.path.join(inp_root, name)
            if "spec" in inp_path:
                continue
            
            opt_path1 = os.path.join(opt_root1, ",".join([str(f0method),name]))
            opt_path2 = os.path.join(opt_root2, ",".join([str(f0method),name])) 
            opt_path3 = os.path.join(opt_root3, ",".join([str(f0method),name]))
            paths.append([inp_path, opt_path1, opt_path2, opt_path3])

        ps = []
        n_p = max(n_p,1)
        for i in range(n_p):
            if device=="cuda":
                featureInput.go(paths[i::n_p])
            else:
                p = Thread(target=featureInput.go,args=(paths[i::n_p],),daemon=True)
                ps.append(p)
                p.start()

        if device != "cuda":
            for p in ps:
                try:
                    p.join()
                except:
                    featureInput.printt("f0_all_fail-%s" % (traceback.format_exc()))
        print(f"Successfully extracted features using {f0method}")
        return True
    except Exception as e:
        print(f"Failed to extract features: {e}")
        return False

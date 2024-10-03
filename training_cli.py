from collections import OrderedDict
import json
import os
import shutil
import traceback
import numpy as np

from tqdm import tqdm
from .lib.audio import SR_MAP
from .lib.train import utils
import datetime
from random import shuffle, randint
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from .lib.infer_pack import commons
from time import sleep
from time import time as ttime
from .lib.train.data_utils import (
    BucketSampler,
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)
from .lib.train.losses import LossBalancer, combined_aux_loss, generator_loss, discriminator_loss, feature_loss, gradient_norm_loss, kl_loss
from .lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

def save_checkpoint(ckpt, name, epoch, hps, model_path=None):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = hps.sample_rate
        opt["f0"] = hps.if_f0
        opt["version"] = hps.version
        if model_path is None: model_path=os.path.join(hps.model_dir,name+".pth")
        torch.save(opt, model_path)
        return "Success."
    except:
        return traceback.format_exc()

class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"

def train_model(hps: "utils.HParams"):
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(8189, 8205+hps.train.num_workers**2))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

    n_gpus = len(hps.gpus.split("-")) if hps.gpus else torch.cuda.device_count()

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    
    gpu_devices = hps.gpus.split("-") if hps.gpus else range(n_gpus)
    

    children = {}
    for i, device in enumerate(gpu_devices):
        subproc = mp.Process(
            target=run,
            args=(
                i,
                n_gpus,
                hps,
                device
            ),
        )
        children[i]=subproc
        subproc.start()

    for i in children:
        children[i].join()

def run(rank, n_gpus, hps, device):
    print(f"{__name__=}")
    global global_step, least_loss, loss_file, best_model_name, gradient_clip_value
    global_step = 0
    loss_file = os.path.join(hps.model_dir,"losses.json")
    gradient_clip_value = np.log(hps.train.c_gp/hps.train.learning_rate) if hps.train.get("c_gp",0.)>0 else None

    if os.path.isfile(loss_file):
        with open(loss_file,"r") as f:
            data: dict = json.load(f)
            least_loss = data.get("least_loss",hps.best_model_threshold)
            best_model_name = data.get("best_model_name","")
    else:
        least_loss = hps.best_model_threshold
        best_model_name = ""

    if hps.version == "v1":
        from .lib.infer_pack.models import (
            SynthesizerTrnMs256NSFsid as RVC_Model_f0,
            SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminator,
        )
    else:
        from .lib.infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
        )

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    if n_gpus>1:
        try:
            dist.init_process_group(backend="gloo", init_method="env://", world_size=n_gpus, rank=rank)
            distributed = True
        except Exception as error:
            print(f"Failed to initialize dist: {error=}")
            distributed = False
    else: distributed=False
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{device}")

    if hps.if_f0:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    
    if distributed:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size * n_gpus,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
    else:
        train_sampler = BucketSampler(
            train_dataset,
            hps.train.batch_size,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s            
            shuffle=True,
        )
    
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=hps.train.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    hps.sync_log_interval(len(train_loader))
    
    if hps.if_f0:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if distributed:
        if torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    try:  # resume training
        _, _, _, epoch_str, d_kwargs = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        if rank == 0: logger.info("loaded D")
        
        _, _, _, epoch_str, g_kwargs = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        if rank == 0: logger.info("loaded G")

        global_step = (epoch_str - 1) * len(train_loader)

    except Exception as e:
        logger.error(f"Failed to load saved pretrains: {e}")
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g,"module"): print(net_g.module.load_state_dict(torch.load(hps.pretrainG, map_location="cpu")["model"]))
            else: print(net_g.load_state_dict(torch.load(hps.pretrainG, map_location="cpu")["model"]))
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            
            if hasattr(net_d,"module"): print(net_d.module.load_state_dict(torch.load(hps.pretrainD, map_location="cpu")["model"]))
            else: print(net_d.load_state_dict(torch.load(hps.pretrainD, map_location="cpu")["model"]))
        d_kwargs = g_kwargs = {}

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    try:
        balancer_state = g_kwargs["balancer"]
        logger.info(f"Using existing balancer: {balancer_state}")
        balancer = LossBalancer(net_g,**balancer_state)
    except Exception as e:
        logger.error(f"Failed to load balancer state: {e}")
        balancer = LossBalancer(
            net_g,            
            weights_decay=.5 / (1 + np.exp(-10 * (epoch_str / hps.total_epoch - 0.16)))+.5, #sigmoid scaled ema .8 at 20% epoch
            loss_decay=.8,
            epsilon=hps.train.eps,
            active=hps.train.get("use_balancer",False),
            use_pareto=hps.train.get("use_pareto",False),
            use_norm=not hps.train.get("fast_mode",False),
            initial_weights=dict(
                loss_gen=1.,
                loss_fm=hps.train.get("c_fm",2.),
                loss_mel=hps.train.get("c_mel",45.),
                loss_kl=hps.train.get("c_kl",1.),
                harmonic_loss=hps.train.get("c_hd",0.),
                tsi_loss=hps.train.get("c_tsi",0.),
                tefs_loss=hps.train.get("c_tefs",0.),
            ))
    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_loader.batch_sampler.set_epoch(epoch)
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
                balancer
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
                balancer
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, _, scaler, loaders, logger, writers, cache, balancer: "LossBalancer"
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, _ = loaders
    if writers is not None:
        writer, _ = writers

    global global_step, least_loss, loss_file, best_model_name, gradient_clip_value

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in tqdm(data_iterator,desc=f"[Epoch {epoch}]: "):
        # Data
        ## Unpack
        if hps.if_f0:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (not hps.if_cache_data_in_gpu) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run: y_hat_mel = y_hat_mel.half()
            wave_orig = wave.clone()
            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            gen_wave = y_hat.detach()
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, gen_wave)
            
            with autocast(enabled=False):
                gradient_penalty = gradient_norm_loss(wave,gen_wave, net_d, eps=hps.train.eps)*hps.train.c_gp if hps.train.get("c_gp",0.)>0 else 0
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc+gradient_penalty

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), gradient_clip_value, batch_size=hps.train.batch_size)
            scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) #*hps.train.get("c_mel",0.)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) #*hps.train.get("c_kl",0.)
                loss_fm = feature_loss(fmap_r, fmap_g) #*hps.train.get("c_fm",0.)
                harmonic_loss, tefs_loss, tsi_loss = combined_aux_loss(
                    wave, y_hat,n_mels=hps.data.n_mel_channels,sample_rate=hps.data.sampling_rate,
                    c_tefs=hps.train.get("c_tefs",0.),
                    c_hd=hps.train.get("c_hd",0.),
                    c_tsi=hps.train.get("c_tsi",0.),
                    n_fft=hps.data.filter_length,
                    hop_length=hps.data.hop_length,
                    win_length=hps.data.win_length,
                    eps=hps.train.eps
                )
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                aux_loss = harmonic_loss + tefs_loss + tsi_loss
                loss_gen_all = balancer.on_train_batch_start(dict(
                    loss_gen=loss_gen,
                    loss_fm=loss_fm,
                    loss_mel=loss_mel,
                    loss_kl=loss_kl,
                    harmonic_loss=harmonic_loss,
                    tsi_loss=tsi_loss,
                    tefs_loss=tefs_loss,
                    ),input=y_hat)
                
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), gradient_clip_value, batch_size=hps.train.batch_size)
            scaler.step(optim_g)
            scaler.update()
        
        if rank == 0:
            if hps.train.log_interval>0 and global_step % hps.train.log_interval == 0: #tensorboard logging
                lr = optim_g.param_groups[0]["lr"]
                logger.info(f"Train Epoch: {epoch} [{100.0 * (epoch-1) / hps.total_epoch:.2f}% complete]")

                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9
                
                scalar_dict = {
                    "total/loss/all": loss_gen_all+loss_disc_all,
                    "total/loss/gen_all": loss_gen_all,
                    "total/loss/aux": aux_loss,
                    "total/loss/disc_all": loss_disc_all,
                    "total/loss/gen": loss_gen,
                    "total/loss/disc": loss_disc,
                    "total/loss/fm": loss_fm,
                    "total/loss/mel": loss_mel,
                    "total/loss/kl": loss_kl,
                    "aux/loss/harmonic": harmonic_loss,
                    "aux/loss/tefs": tefs_loss,
                    "aux/loss/tsi": tsi_loss,
                    "gradient/lr": lr,
                    "gradient/grad_norm_disc": grad_norm_d,
                    "gradient/grad_norm_gen": grad_norm_g,
                    "gradient/gradient_penalty": gradient_penalty,
                    **{f"loss/g/{i}": v for i, v in enumerate(losses_gen)},
                    **{f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)},
                    **{f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                }

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "slice/diff": utils.plot_spectrogram_to_numpy((y_mel[0]-y_hat_mel[0]).abs().data.cpu().numpy(), cmap="hot")
                }

                with torch.no_grad():
                    if hasattr(net_g, "module"): inference = net_g.module.infer
                    else: inference = net_g.infer
                    if hps.if_f0: wave_gen = inference(phone, phone_lengths, pitch, pitchf, sid)[0][0, 0].data
                    else: wave_gen = inference(phone, phone_lengths, sid)[0][0, 0].data
                    
                audio_dict = {
                    "slice/wave_org": wave_orig[0][0],
                    "slice/wave_gen": wave_gen
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                    audios=audio_dict,
                    audio_sampling_rate=SR_MAP[hps.sample_rate]
                )
        global_step += 1
    # /Run steps

    if hps.save_every_epoch>0 and (epoch % hps.save_every_epoch == 0) and rank == 0:
            
        saved_epoch = 23333 if hps.if_latest else epoch
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, f"G_{saved_epoch}.pth"),
            balancer=balancer.to_dict()
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, f"D_{saved_epoch}.pth"),
        )
        if hps.save_every_weights:
            ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            save_name = f"{hps.name}_e{epoch}_s{global_step}"
            status = save_checkpoint(ckpt,save_name,epoch,hps)
            logger.info(f"saving ckpt {save_name}: {status}")

    if rank == 0:
        total_loss = balancer.weighted_ema_loss + loss_disc_all
        logger.info(f"====> Epoch {epoch} ({total_loss=:.3f}): {global_step=} {lr=:.2E} {epoch_recorder.record()}")
        logger.info(f"|| {loss_disc_all=:.3f}: {loss_disc=:.3f}, {gradient_penalty=:.3f}")
        logger.info(f"|| {loss_gen_all=:.3f}: {loss_gen=:.3f}, {loss_fm=:.3f}, {loss_mel=:.3f}, {loss_kl=:.3f}")
        logger.info(f"|| {harmonic_loss=:.3f}, {tefs_loss=:.3f}, {tsi_loss=:.3f}")
        balancer.on_epoch_end(.5 / (1 + np.exp(-10 * (epoch / hps.total_epoch - 0.16)))+.5) #sigmoid scaling of ema

        if total_loss<least_loss:
            least_loss = total_loss
            logger.info(f"[lowest loss] {least_loss=:.3f}: {loss_disc=:.3f} {gradient_penalty=:.3f} <==> {loss_gen=:.3f} {loss_fm=:.3f} {loss_mel=:.3f} {loss_kl=:.3f}")
            logger.info(f"[aux] {aux_loss=:.3f}: {harmonic_loss=:.3f}, {tefs_loss=:.3f}, {tsi_loss=:.3f}")

            if hps.save_best_model:
                if hasattr(net_g, "module"): ckpt = net_g.module.state_dict()
                else: ckpt = net_g.state_dict()
                
                best_model_name = f"{hps.name}_e{epoch}_s{global_step}_loss{least_loss:.0f}" if hps.save_every_weights else f"{hps.name}_loss{least_loss:2.0f}"
                status = save_checkpoint(ckpt,best_model_name,epoch,hps)
                logger.info(f"=== saving best model {best_model_name}: {status=} ===")
            
            with open(loss_file,"w") as f:
                json.dump(dict(least_loss=least_loss.item(),best_model_name=best_model_name,epoch=epoch,steps=global_step,
                                loss_weights = dict(balancer.ema_weights),
                                scalar_dict={
                                    "total/loss/all": commons.serialize_tensor(loss_gen_all+loss_disc_all),
                                    "total/loss/gen_all": commons.serialize_tensor(loss_gen_all),
                                    "total/loss/aux": commons.serialize_tensor(aux_loss),
                                    "total/loss/disc_all": commons.serialize_tensor(loss_disc_all),
                                    "total/loss/gen": commons.serialize_tensor(loss_gen),
                                    "total/loss/disc": commons.serialize_tensor(loss_disc),
                                    "total/loss/fm": commons.serialize_tensor(loss_fm),
                                    "total/loss/mel": commons.serialize_tensor(loss_mel),
                                    "total/loss/kl": commons.serialize_tensor(loss_kl),
                                    "aux/loss/harmonic": commons.serialize_tensor(harmonic_loss),
                                    "aux/loss/tefs": commons.serialize_tensor(tefs_loss),
                                    "aux/loss/tsi": commons.serialize_tensor(tsi_loss),
                                    "gradient/grad_norm_disc": commons.serialize_tensor(grad_norm_d),
                                    "gradient/grad_norm_gen": commons.serialize_tensor(grad_norm_g),
                                    "gradient/gradient_penalty": commons.serialize_tensor(gradient_penalty),
                                }),f,indent=2)

    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
        if hps.save_best_model and os.path.isfile(loss_file):
            with open(loss_file,"r") as f:
                data = json.load(f)
                best_model_name = data.get("best_model_name","")
                best_model_path = os.path.join(hps.model_dir,f"{best_model_name}.pth")
                if os.path.isfile(best_model_path):
                    shutil.copy(best_model_path,os.path.join(
                        os.path.dirname(hps.model_path),
                        f"{os.path.basename(hps.model_path).split('.')[0]}-lowest.pth"))

        status = save_checkpoint(ckpt,hps.name,epoch,hps,model_path=hps.model_path)
        logger.info(f"saving final ckpt {hps.model_path}: {status}")
        sleep(1)
        os._exit(0)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    hps = utils.get_hparams()
    train_model(hps)
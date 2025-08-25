import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
from colorlog import ColoredFormatter
import tqdm
from itertools import chain
import random
import numpy as np
from pathlib import Path
from einops import rearrange
import time
import json

try:
    import lpips
except:
    raise Exception("Need lpips to valid.")
from torchmetrics import StructuralSimilarityIndexMeasure

import sys
sys.path.append(".")
from model.ema_model import EMA
from dataset.ddp_sampler import CustomDistributedSampler
from dataset.video_dataset import TrainVideoDataset, ValidVideoDataset
from utils.module_utils import resolve_str_to_obj
from utils.video_utils import tensor_to_video
from diffusers_vae.src.diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers_vae.src.diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLHunyuanVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLCogVideoX

feature_corresponding = ["mid_block","up_block_0","up_block_1","up_block_2","up_block_3"]

teacher_model_mapping = {"CogVideoX":AutoencoderKLCogVideoX, "Hunyuan":AutoencoderKLHunyuanVideo,
                         "LTX":AutoencoderKLLTXVideo}

def normalize_tensor_to_0_1(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    epsilon = 1e-10
    normalized_tensor = (tensor - min_val) / (max_val - min_val + epsilon)
    
    return normalized_tensor

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank,ckpt_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        f"[rank{rank}] %(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        reset=True,
        style="%",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(ckpt_dir,'log.txt'))  
    file_handler.setLevel(logging.DEBUG)  
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger

def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)


def get_exp_name(args):
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}"

def set_train(modules):
    for module in modules:
        module.train()

def set_eval(modules):
    for module in modules:
        module.eval()

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)

def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath


def valid(global_rank, rank, teacher_model, model, val_dataloader, precision, args):
    if args.eval_lpips:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(rank)

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    ssim_list  =  []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank)
            with torch.cuda.amp.autocast(dtype=precision):
                latents = teacher_model.encode(inputs).latent_dist.sample()
                video_recon = model.module.decode(latents,return_dict=False)[0]

            inputs = normalize_tensor_to_0_1(rearrange(inputs, "b c t h w -> (b t) c h w")).contiguous()
            video_recon = normalize_tensor_to_0_1(rearrange(
                torch.clamp(video_recon,-1.0,1.0), "b c t h w -> (b t) c h w"
            )).contiguous()

            # Calculate PSNR
            mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))
            psnr = 20 * torch.log10(1 / torch.sqrt(mse))
            psnr = psnr.mean().detach().cpu().item()

            # Calculate LPIPS
            if args.eval_lpips:
                lpips_score = (
                    lpips_model.forward(inputs, video_recon)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                lpips_list.append(lpips_score)
            
            ssim_value = ssim(inputs, video_recon).mean().detach().cpu().item()
            ssim_list.append(ssim_value)

            psnr_list.append(psnr)
            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()
    return psnr_list, lpips_list, ssim_list


def gather_valid_result(psnr_list, lpips_list, ssim_list, rank, world_size):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_ssim_list = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_ssim_list, ssim_list)
    return (
        np.array(gathered_psnr_list).mean(),
        np.array(gathered_lpips_list).mean(),
        np.array(gathered_ssim_list).mean(),
    )

def load_json_to_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
        return data_dict
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def train(args):
    # Setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()

    # Init
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
            with open(os.path.join(ckpt_dir,'log.txt'), 'w') as file:
                file.write('log.txt create\n')
            print("log.txt create\n")
        except:
            print(f"`{ckpt_dir}` exists!")
            time.sleep(5)
    if global_rank == 0:
        logger = setup_logger(rank,ckpt_dir)
    dist.barrier()

    if args.model_config:
        model_config = load_json_to_dict(args.model_config)

    model = AutoencoderKLTurboVAED.from_config(
        config=model_config
    ) 
    
    if args.pretrained_model_name_or_path is not None:
        if global_rank == 0:
            logger.warning(
                f"You are loading a checkpoint from `{args.pretrained_model_name_or_path}`."
            )
        checkpoint = torch.load(args.pretrained_model_name_or_path, map_location="cpu")
        model.decoder.load_state_dict(checkpoint, strict=False)
    else:
        if global_rank == 0:
            logger.warning(f"Model will be inited randomly.")
    
    if args.teacher_model_name is not None and args.teacher_pretrained_model_name_or_path is not None:
        teacher_model = teacher_model_mapping[args.teacher_model_name].from_pretrained(
            args.teacher_pretrained_model_name_or_path
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    if global_rank == 0:
        model_config = dict(**model.config)
        args_config = dict(**vars(args))

        logger.info(f"{model_config}")
        logger.info(f"{args_config}")

    dist.barrier()
    
    # Load discriminator model
    disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
    if global_rank == 0:
        logger.warning(
            f"disc_class: {args.disc_cls} perceptual_weight: {args.perceptual_weight}  loss_type: {args.loss_type}"
        )
    disc = disc_cls(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type
    )

    # DDP
    model = model.to(rank, )
    model = DDP(
        model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    disc = disc.to(rank)
    disc = DDP(
        disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    if args.teacher_pretrained_model_name_or_path is not None:
        teacher_model = teacher_model.to(rank, )

    # Load dataset
    dataset = TrainVideoDataset(
        args.video_path,
        sequence_length=args.num_frames,
        resolution=args.resolution,
        sample_rate=args.sample_rate,
        dynamic_sample=args.dynamic_sample,
        cache_file="idx.pkl",
        is_main_process=global_rank == 0,
    )
    ddp_sampler = CustomDistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=args.dataset_num_worker,
    )
    val_dataset = ValidVideoDataset(
        real_video_dir=args.eval_video_path,
        num_frames=args.eval_num_frames,
        sample_rate=args.eval_sample_rate,
        crop_size=args.eval_resolution,
        resolution=args.eval_resolution,
    )
    indices = range(args.eval_subset_size)
    val_dataset = Subset(val_dataset, indices=indices)
    val_sampler = CustomDistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    # Optimizer
    modules_to_train = [model.module.decoder]
    if model_config["aligned_feature_projection_dim"]:
        modules_to_train += [model.module.aligned_feature_projection_heads]

    # if not args.freeze_encoder:
    #     modules_to_train += [model.module.encoder]
    # else:
    #     model.module.encoder.eval()
    #     model.module.encoder.requires_grad_(False)
    #     if global_rank == 0:
    #         logger.warning("Encoder is freezed!")

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(filter(lambda p: p.requires_grad, module.parameters()))

    gen_optimizer = torch.optim.AdamW(
        parameters_to_train, lr=args.lr, betas=(args.betas_1,args.betas_2),
        weight_decay=args.weight_decay,eps=args.eps
    )
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), 
        lr=args.disc_lr, weight_decay=args.weight_decay, eps=args.eps, betas=(args.betas_1,args.betas_2)
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    print(precision)
    
    # Load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=True)
        disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
        disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        if global_rank == 0:
            logger.info(
                f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
            )
    start_step = current_step
    if args.ema:
        if global_rank == 0:
            logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()

    # Training loop
    if global_rank == 0:
        logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        logger.info(f"=== Model Params ===")
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"===========")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")

    # Training Bar
    bar_desc = ""
    bar = None
    if global_rank == 0:
        args.max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=args.max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {args.max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    # Training Loop
    num_epochs = args.epochs

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()

    for epoch in range(num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["video"].to(rank)
            current_step += 1

            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False

            assert (
                step_gen or step_dis
            ), "You should backward either Gen or Dis in a step."

            with torch.cuda.amp.autocast(dtype=precision):
                posterior = teacher_model.encode(inputs).latent_dist
                latents = posterior.sample()
                recon, student_feature = model.module.decode(latents,feature_enabled=True)
                with torch.no_grad():
                    teacher_recon, teacher_feature = teacher_model.decode(latents,feature_enabled=True)
            
            # Generator Step
            if step_gen:
                with torch.cuda.amp.autocast(dtype=precision):
                    g_loss, g_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=0,
                        global_step=current_step,
                        last_layer=model.module.get_last_layer(),
                        split="train",
                    )
                    feature_distillation_loss = 0.0
                    for index in args.feature_indices:
                        cur_layer = feature_corresponding[index]
                        if model_config["aligned_blks_indices"] is not None and index in model_config["aligned_blks_indices"]:
                            layer_i = model_config["aligned_blks_indices"].index(index)

                            projection_head = model.module.aligned_feature_projection_heads[layer_i]
                            current_student_feature = projection_head(student_feature[cur_layer])

                            # upsampling align h w
                            batch_size, num_channels, num_frames, height, width = current_student_feature.shape
                            current_teacher_feature_channels = teacher_feature[cur_layer].shape[1]

                            if num_channels == current_teacher_feature_channels*4 : # spatial upsampling
                                current_student_feature = current_student_feature.permute(0, 2, 1, 3, 4)
                                current_student_feature = current_student_feature.reshape(batch_size * num_frames, num_channels, height, width)
                                current_student_feature = torch.nn.functional.pixel_shuffle(current_student_feature, 2) 
                                _, c, h, w = current_student_feature.shape
                                current_student_feature = current_student_feature.reshape(batch_size, num_frames, c, h, w)
                                current_student_feature = current_student_feature.permute(0, 2, 1, 3, 4)
                            elif num_channels == current_teacher_feature_channels*8: # spatial-temporal upsampling
                                current_student_feature = current_student_feature.reshape(
                                    batch_size, -1, 2, 2, 2, num_frames, height, width
                                )
                                current_student_feature = current_student_feature.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
                                current_student_feature = current_student_feature[:, :, 1 :]

                            feature_distillation_loss += torch.nn.functional.mse_loss(
                                current_student_feature,
                                teacher_feature[cur_layer].squeeze(1).to(rank).detach()
                            )
                        else:
                            feature_distillation_loss += torch.nn.functional.mse_loss(
                                student_feature[cur_layer],
                                teacher_feature[cur_layer].squeeze(1).to(rank).detach()
                            )
                    g_loss += args.distillation_weight * feature_distillation_loss
                    # soft_distillation_loss = torch.nn.functional.mse_loss(recon,teacher_recon)
                    # g_loss += args.distillation_weight * soft_distillation_loss
                g_loss = g_loss / args.accumulation_steps
                scaler.scale(g_loss).backward()
                # scaler.unscale_(gen_optimizer)
                # torch.nn.utils.clip_grad_norm_(parameters_to_train, 5e6)
                if current_step % args.accumulation_steps == 0:
                    scaler.step(gen_optimizer)
                    scaler.update()
                    gen_optimizer.zero_grad()
                    if args.ema:
                        ema.update()
                if global_rank == 0 and current_step % args.log_steps == 0:
                    logger.info(f"===========")
                    logger.info(
                        f"train/generator_loss in {current_step}: {g_loss.item()}" 
                    )
                    logger.info(
                        f"train/rec_loss in {current_step}: {g_log['train/rec_loss']}"
                    )
                    logger.info(
                        f"train/kl_loss in {current_step}: {g_log['train/kl_loss']}"
                    )
                    logger.info(
                        f"train/feature_distillation_loss in {current_step}: {feature_distillation_loss}"
                    )


            # Discriminator Step
            if step_dis:
                with torch.cuda.amp.autocast(dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=current_step,
                        last_layer=None,
                        split="train",
                    )
                disc_optimizer.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(disc.module.discriminator.parameters(), 1.0)
                scaler.step(disc_optimizer)
                scaler.update()
                if global_rank == 0 and (current_step+1) % args.log_steps == 0:
                    logger.info(
                        f"train/discriminator_loss in {current_step} : {d_loss.item()}"
                    )

            update_bar(bar)

            def valid_model(model, name=""):
                set_eval(modules_to_train)
                psnr_list, lpips_list, ssim_list = valid(
                    global_rank, rank, teacher_model, model, val_dataloader, precision, args
                )
                valid_psnr, valid_lpips, valid_ssim = gather_valid_result(
                    psnr_list, lpips_list, ssim_list, rank, dist.get_world_size()
                )
                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    logger.info({f"val{name}/psnr in {current_step}:{valid_psnr}"})
                    logger.info({f"val{name}/lpips in {current_step}:{valid_lpips}"})
                    logger.info({f"val{name}/ssim in {current_step}:{valid_ssim}"})
                    logger.info(f"{name} Validation done.")

            if current_step % args.eval_steps == 0 or current_step == start_step+1 or current_step == args.max_steps:
                if global_rank == 0:
                    logger.info("Starting validation...")
                valid_model(model)
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, "ema")
                    ema.restore()

            # Checkpoint
            if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "dics_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else {},
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help=""
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of steps to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--ckpt_dir", type=str, default="./outputs/", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="accumulation steps for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--disc_lr", type=float, default=1e-5, help="discriminator learning rate")
    parser.add_argument("--betas_1", type=float, default=0.9, help="betas 1")
    parser.add_argument("--betas_2", type=float, default=0.999, help="betas 2")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="optimizer eps")

    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="")

    # Data
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--latent_path", type=str, default=None, help="")
    parser.add_argument("--feature_path", type=str, default=None, help="")

    parser.add_argument("--num_frames", type=int, default=17, help="")
    parser.add_argument("--resolution", type=int, default=256, help="")
    parser.add_argument("--sample_rate", type=int, default=2, help="")
    parser.add_argument("--dynamic_sample", action="store_true", help="")
    # Generator model
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help=""
    )
    parser.add_argument(
        "--teacher_model_name", type=str, default=None, help=""
    )
    parser.add_argument(
        "--teacher_pretrained_model_name_or_path", type=str, default=None, help=""
    )

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument("--not_resume_training_process", action="store_true", help="")
    parser.add_argument("--model_config", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    # Discriminator Model
    parser.add_argument("--load_disc_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="losses.LPIPSWithDiscriminator3D",
        help="",
    )
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")

    parser.add_argument("--distillation_weight", type=float, default=1.0, help="")
    parser.add_argument('--feature_indices', nargs='*', type=int, default=[], help='feature align')

    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_video_path", type=str, default=None, help="")
    parser.add_argument("--eval_latent_path", type=str, default=None, help="")
    parser.add_argument("--eval_num_frames", type=int, default=17, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_sample_rate", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=2, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

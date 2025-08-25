import json
import torch
import tqdm
import os
import numpy as np
from einops import rearrange
import scipy
import random
try:
    import lpips
except:
    raise Exception("Need lpips to valid.")
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.append(".")
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchmetrics import StructuralSimilarityIndexMeasure
from dataset.video_dataset import ValidVideoDataset
from dataset.ddp_sampler import CustomDistributedSampler
from diffusers_vae.src.diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLHunyuanVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLCogVideoX

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

def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]
    return mu, sigma

def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray):
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)

# download from "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
i3d_path = "./checkpoints/i3d_torchscript.pt"

def valid(global_rank, rank, teacher_model, model, val_dataloader):
    lpips_model = lpips.LPIPS(net="alex", spatial=True)
    lpips_model.to(rank)
    lpips_model = DDP(lpips_model, device_ids=[rank])
    lpips_model.requires_grad_(False)
    lpips_model.eval()

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(rank)
    detector = torch.jit.load(i3d_path).eval().to("cuda")

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    ssim_list  =  []
    feats_real = []
    feats_fake = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                latents = teacher_model.encode(inputs).latent_dist.sample()
                video_recon = model.decode(latents,return_dict=False)[0]

            inputs = normalize_tensor_to_0_1(inputs).contiguous()
            video_recon = normalize_tensor_to_0_1(torch.clamp(video_recon,-1.0,1.0)).contiguous()

            with torch.no_grad():
                micro_feats_real = (
                    detector(inputs, rescale=False, resize=True, return_features=True)
                    .cpu()
                    .numpy()
                )
                micro_feats_fake = (
                    detector(video_recon, rescale=False, resize=True, return_features=True)
                    .cpu()
                    .numpy()
                )
            feats_real.append(micro_feats_real)
            feats_fake.append(micro_feats_fake)

            inputs = (rearrange(inputs, "b c t h w -> (b t) c h w")).contiguous()
            video_recon = (rearrange(
                video_recon, "b c t h w -> (b t) c h w"
            )).contiguous()

            # Calculate PSNR
            mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))
            psnr = 20 * torch.log10(1 / torch.sqrt(mse))
            psnr = psnr.mean().detach().cpu().item()

            # Calculate LPIPS
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
        feats_real = np.concatenate(feats_real, axis=0)
        feats_fake = np.concatenate(feats_fake, axis=0)
        rfvd = compute_fvd(feats_real, feats_fake)
        print(f"rFVD:{rfvd}")
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

teacher_model = AutoencoderKLHunyuanVideo.from_pretrained(
    ""
).to("cuda")

model_config = ""
resume_from_checkpoint = ""
model = AutoencoderKLTurboVAED.from_config(
    config=load_json_to_dict(model_config)
)
checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
model.decoder.load_state_dict(checkpoint)
model = model.to("cuda")

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

eval_video_path = "UCF-101-val"
eval_subset_size = 3783

val_dataset = ValidVideoDataset(
    real_video_dir=eval_video_path,
    num_frames=17,
    sample_rate=1,
    crop_size=256,
    resolution=256,
)
indices = range(eval_subset_size)
val_dataset = Subset(val_dataset, indices=indices)
val_sampler = CustomDistributedSampler(val_dataset)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    sampler=val_sampler,
    pin_memory=True,
)

psnr_list, lpips_list, ssim_list = valid(
    0, 0, teacher_model, model, val_dataloader
)
valid_psnr, valid_lpips, valid_ssim = gather_valid_result(
    psnr_list, lpips_list, ssim_list, 0, dist.get_world_size()
)

print(f"PSNR:{valid_psnr}")
print(f"LPIPS:{valid_lpips}")
print(f"SSIM:{valid_ssim}")

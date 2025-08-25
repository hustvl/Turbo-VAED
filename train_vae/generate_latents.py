import torch
from torchvision import transforms
from decord import VideoReader, cpu
from einops import rearrange
from torchvision.io import write_video
import numpy as np
import os
from dataset.transform import CenterCropVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLHunyuanVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLCogVideoX

vae = AutoencoderKLHunyuanVideo.from_pretrained(
    ""
).to("cuda")

transform = transforms.Compose([
    transforms.Resize(256),
    CenterCropVideo(256)
])

video_root_path = ""
latents_save_path = ""

def save_latent(root, video_name):
    video_path = os.path.join(root, video_name)
    video_reader = VideoReader(video_path,ctx=cpu(0))
    fps = video_reader.get_avg_fps()

    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
    
    if len(video) == 0:
        print(f"Skipping {video_name}: No frames found.")
        return
    
    video = rearrange(torch.tensor(video),'t h w c -> t c h w')
    video = transform(video)
    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0)
    video = video / 127.5 - 1.0
    video= video[:,:,:17,:,:]

    if video.nelement() == 0:
        print(f"Skipping {video_name}: Resulting tensor is empty after processing.")
        return

    video = video.to(device="cuda", dtype=vae.dtype)
    with torch.no_grad():
        latents = vae.encode(video).latent_dist.sample()
    latents = latents.to('cpu',dtype=vae.dtype)
    latents_path = os.path.join(latents_save_path,f"{os.path.splitext(video_name)[0]}.pt")
    print(f'Saving latents to {latents_path}')
    torch.save(latents,latents_path)

for root, dirs, files in os.walk(video_root_path):
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    for file in files:
        if file.lower().endswith(video_extensions):
            save_latent(root, file)

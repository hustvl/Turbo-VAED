import torch
from PIL import Image
from torchvision import transforms
import json
from decord import VideoReader, cpu
from einops import rearrange
from torchvision.io import write_video
import os
from diffusers_vae.src.diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLHunyuanVideo
from diffusers_vae.src.diffusers.models.autoencoders import AutoencoderKLCogVideoX

teacher_model_mapping = {"CogVideoX":AutoencoderKLCogVideoX, "Hunyuan":AutoencoderKLHunyuanVideo,
                         "LTX":AutoencoderKLLTXVideo}

def load_json_to_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
        return data_dict
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

model_config = ""
resume_from_checkpoint = ""
video_root_path = ""
save_path = ""
teacher_model_name = ""

teacher_model = teacher_model_mapping[teacher_model_name].from_pretrained(
    ""
).to("cuda")

model = AutoencoderKLTurboVAED.from_config(
    config=load_json_to_dict(model_config)
)
checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
model.decoder.load_state_dict(checkpoint, strict=False)
model = model.to("cuda")

transform = transforms.Compose([
    transforms.Resize(size=(512,512))
])

def transform_video(video_name):
    video_path = os.path.join(video_root_path,video_name)
    video_reader = VideoReader(video_path,ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
    video = rearrange(torch.tensor(video),'t h w c -> t c h w')
    video = transform(video)
    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0)
    video = video / 127.5 - 1.0
    video = video[:,:,:17,:,:]
    video = video.to(device="cuda", dtype=model.dtype)

    with torch.no_grad():
        latents = teacher_model.encode(video).latent_dist.sample()
        results = model.decode(latents,return_dict=False)[0]

    results = rearrange(results.squeeze(0), 'c t h w -> t h w c')
    results = (torch.clamp(results,-1.0,1.0) + 1.0) * 127.5
    results = results.to('cpu', dtype=torch.uint8)

    write_video(save_path, results,fps=int(round(fps)),options={'crf': '10'})

for root, dirs, files in os.walk(video_root_path):
  for name in files:
    transform_video(name)
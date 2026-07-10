#!/usr/bin/env python
"""Reconstruct one video with the Wan teacher and Turbo-VAED decoders."""

import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import CenterCrop, Resize

from wan_decoder_utils import (
    DEFAULT_TURBO_CONFIG,
    DEFAULT_WAN_MODEL,
    load_turbo_decoder,
    load_wan_teacher,
    resolve_dtype,
)


def load_video(path, num_frames, resolution, start_frame):
    if start_frame < 0:
        raise ValueError("--start_frame must be non-negative")

    end_frame = start_frame + num_frames
    decoded_frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_rate = float(stream.average_rate) if stream.average_rate is not None else 24.0
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index < start_frame:
                continue
            if frame_index >= end_frame:
                break
            decoded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))

    if len(decoded_frames) != num_frames:
        raise ValueError(
            f"Decoded {len(decoded_frames)} frames, but {num_frames} frames "
            f"starting at frame {start_frame} were requested."
        )

    frames = torch.stack(decoded_frames).permute(0, 3, 1, 2)
    frames = Resize(resolution, antialias=True)(frames)
    frames = CenterCrop(resolution)(frames)
    video = rearrange(frames, "t c h w -> 1 c t h w").float() / 127.5 - 1.0
    return video, frame_rate


def video_to_uint8(video):
    frames = rearrange(video[0], "c t h w -> t h w c")
    return ((frames.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu()


def add_labels(frames, left_label, right_label):
    frames_np = frames.numpy()
    output = []
    font = ImageFont.load_default(size=11)
    for frame in frames_np:
        canvas = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        half_width = frame.shape[1] // 2

        for x, label in ((6, left_label), (half_width + 6, right_label)):
            y = 5
            text_box = draw.textbbox((x, y), label, font=font)
            draw.rectangle(
                (text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
                fill=(0, 0, 0, 150),
            )
            draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)

        draw.line((half_width, 0, half_width, canvas.height - 1), fill=(255, 255, 255, 190), width=1)
        canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
        output.append(np.asarray(canvas))
    return torch.from_numpy(np.stack(output)).to(torch.uint8)


def save_video(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_rate = Fraction(str(fps)).limit_denominator(1000)
    frames_np = frames.numpy()

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=frame_rate)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "10", "preset": "slow"}

        for frame_array in frames_np:
            video_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Wan2.2-TI2V-5B and Turbo-VAED reconstructions on one input video."
    )
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_model", default=DEFAULT_WAN_MODEL)
    parser.add_argument("--model_config", default=str(DEFAULT_TURBO_CONFIG))
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--output_dir", default="outputs/wan_demo")
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None, help="Defaults to the input video's FPS.")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the Wan teacher VAE.")

    device = torch.device("cuda")
    dtype = resolve_dtype(args.precision)
    video, input_fps = load_video(
        args.input_video,
        num_frames=args.num_frames,
        resolution=args.resolution,
        start_frame=args.start_frame,
    )
    video = video.to(device=device, dtype=dtype)

    teacher = load_wan_teacher(args.teacher_model, device=device, dtype=dtype)
    turbo, loaded_from = load_turbo_decoder(
        args.model_config,
        args.checkpoint,
        args.weights,
        device=device,
        dtype=dtype,
    )
    print(f"[MODEL] Turbo-VAED loaded from {loaded_from}")

    with torch.inference_mode():
        latents = teacher.encode(video).latent_dist.mode()
        teacher_recon = teacher.decode(latents, return_dict=False)[0]
        turbo_recon = turbo.decode(latents, return_dict=False)[0]

    teacher_frames = video_to_uint8(teacher_recon)
    turbo_frames = video_to_uint8(turbo_recon)
    if teacher_frames.shape != turbo_frames.shape:
        raise RuntimeError(
            f"Decoder output shapes do not match: teacher={teacher_frames.shape}, "
            f"Turbo-VAED={turbo_frames.shape}"
        )

    comparison = torch.cat([teacher_frames, turbo_frames], dim=2)
    comparison = add_labels(comparison, "Wan2.2-TI2V-5B", "Turbo-VAED")
    output_dir = Path(args.output_dir)
    output_fps = args.fps or input_fps
    save_video(output_dir / "wan_2_2_5b_t2v.mp4", teacher_frames, output_fps)
    save_video(output_dir / "turbo_vaed.mp4", turbo_frames, output_fps)
    save_video(output_dir / "comparison.mp4", comparison, output_fps)

    print(f"[INPUT] {args.input_video}")
    print(f"[LATENT] shape={tuple(latents.shape)}, dtype={latents.dtype}")
    print(f"[OUTPUT] {output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Shared loading helpers for the Wan Turbo-VAED release."""

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = PROJECT_ROOT / "diffusers_vae" / "src"
if str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import (  # noqa: E402
    AutoencoderKLTurboVAED,
)


DEFAULT_WAN_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_TURBO_CONFIG = PROJECT_ROOT / "configs" / "Turbo-VAED-Wan.json"


def resolve_dtype(precision):
    dtypes = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return dtypes[precision]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict) and "gen_model" in state_dict:
            return state_dict["gen_model"]
        return state_dict
    if isinstance(checkpoint, dict) and "gen_model" in checkpoint:
        return checkpoint["gen_model"]
    return checkpoint


def _strip_module_prefix(state_dict):
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_wan_teacher(model_path, device, dtype):
    model = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_turbo_decoder(config_path, checkpoint_path, weights, device, dtype):
    model = AutoencoderKLTurboVAED.from_config(config=load_json(config_path))
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )

    raw_state = _strip_module_prefix(_unwrap_checkpoint_state(checkpoint))
    model.load_state_dict(raw_state, strict=True)
    loaded_from = "state_dict.gen_model"

    if weights == "ema":
        ema_state = checkpoint.get("ema_state_dict") if isinstance(checkpoint, dict) else None
        if not ema_state:
            raise KeyError(f"`ema_state_dict` not found in checkpoint: {checkpoint_path}")
        ema_state = _strip_module_prefix(ema_state)
        model_state = model.state_dict()
        matched = 0
        for key, value in ema_state.items():
            if key in model_state:
                model_state[key] = value
                matched += 1
        model.load_state_dict(model_state, strict=True)
        loaded_from = f"state_dict.gen_model + ema_state_dict ({matched} tensors)"

    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model, loaded_from

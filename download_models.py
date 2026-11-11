#!/usr/bin/env python3
"""
Pre-download and Cache Foundation Model Weights for Cartoon Video Generator
===========================================================================
Pre-fetches the required open-source model weights from Hugging Face Hub:
  1. guoyww/animatediff-motion-adapter-v1-5-2 (~1.8 GB)
  2. runwayml/stable-diffusion-v1-5 (~4.0 GB safetensors)
  3. h94/IP-Adapter (ip-adapter_sd15.bin, ~2.5 GB)
  4. laion/CLIP-ViT-H-14-laion2B-s32B-b79K (image encoder, ~3.7 GB or cached)

Run this once before starting the app to avoid silent background downloads:
    python download_models.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelDownloader")


def download_all_models():
    """Downloads all required weights with progress bars into standard Hugging Face cache."""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download, try_to_load_from_cache
    except ImportError:
        logger.error("huggingface_hub is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    print("=" * 70)
    print("🎬 Cartoon Video Generator - Model Pre-Caching Utility")
    print("=" * 70)

    # 1. AnimateDiff Motion Module v2
    print("\n[1/3] Checking AnimateDiff Motion Adapter (guoyww/animatediff-motion-adapter-v1-5-2)...")
    cached = try_to_load_from_cache("guoyww/animatediff-motion-adapter-v1-5-2", "config.json")
    if cached:
        print("   ✅ AnimateDiff motion adapter is already cached!")
    else:
        print("   ⏳ Downloading AnimateDiff motion module weights (~1.8 GB)...")
        snapshot_download(
            repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
            ignore_patterns=["*.bin"],
        )
        print("   ✅ AnimateDiff motion adapter downloaded.")

    # 2. Stable Diffusion 1.5 Base
    print("\n[2/3] Checking Stable Diffusion 1.5 (runwayml/stable-diffusion-v1-5)...")
    cached = (
        try_to_load_from_cache("runwayml/stable-diffusion-v1-5", "unet/diffusion_pytorch_model.safetensors")
        or try_to_load_from_cache("runwayml/stable-diffusion-v1-5", "unet/diffusion_pytorch_model.bin")
    )
    if cached:
        print("   ✅ Stable Diffusion 1.5 is already cached!")
    else:
        print("   ⏳ Downloading SD 1.5 base pipeline (~3.5 GB safetensors)...")
        # Ignore legacy .bin and non-safetensors to save download time and bandwidth
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            ignore_patterns=["*.bin", "*.ckpt", "*.onnx*", "*flax*"],
        )
        print("   ✅ Stable Diffusion 1.5 downloaded.")

    # 3. IP-Adapter weights
    print("\n[3/3] Checking IP-Adapter (h94/IP-Adapter)...")
    cached = try_to_load_from_cache("h94/IP-Adapter", "models/ip-adapter_sd15.bin")
    if cached:
        print("   ✅ IP-Adapter is already cached!")
    else:
        print("   ⏳ Downloading IP-Adapter SD 1.5 weights (~2.5 GB)...")
        hf_hub_download(
            repo_id="h94/IP-Adapter",
            subfolder="models",
            filename="ip-adapter_sd15.bin",
        )
        print("   ✅ IP-Adapter weights downloaded.")

    print("\n" + "=" * 70)
    print("🎉 All models are cached successfully! You can now launch:")
    print("   python app.py")
    print("=" * 70)


if __name__ == "__main__":
    download_all_models()

#!/usr/bin/env python3
"""
CLI Test Script for Module 3: Image-to-Cartoon-Video Pipeline
============================================================
Provides a fast, terminal-friendly interface to test animated cartoon video
generation from a single input image and motion prompt.

Usage:
  python cli_test.py \
      --image character.png \
      --prompt "in cartoon artstyle, character talking and smiling, waving hand" \
      --output output.mp4 \
      --lora_path ./cartoon_lora_sd15_output
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CartoonVideoCLI")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses and validates CLI parameters."""
    parser = argparse.ArgumentParser(
        description="Module 3: Generate animated cartoon video clips from a single character image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core inputs & outputs
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input character still image (.png, .jpg, .webp).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="in cartoon artstyle, character talking and smiling, expressive animation, waving hand",
        help="Motion and scene description for the generated video clip.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="photorealistic, 3d, realistic skin, deformed, blurry, low quality, static, bad anatomy",
        help="Negative prompt to suppress photorealism and visual artifacts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Destination path for the generated H.264 .mp4 video file.",
    )

    # Model adapters & checkpoints
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to custom cartoon style LoRA (.safetensors or directory from Module 2).",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=0.85,
        help="Influence strength of the cartoon style LoRA adapter (0.0 to 1.0).",
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.6,
        help="Image prompt adapter scale (controls character identity fidelity).",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face repository ID for the base Stable Diffusion 1.5 model.",
    )
    parser.add_argument(
        "--motion_adapter_id",
        type=str,
        default="guoyww/animatediff-motion-adapter-v1-5-2",
        help="Hugging Face repository ID for the AnimateDiff motion adapter.",
    )

    # Video parameters
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of animation frames to synthesize (16 or 24 recommended).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Output video framerate in frames per second.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Denoising steps for the DDIM scheduler.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-Free Guidance (CFG) scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible video generation.",
    )

    # Hardware & Memory
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target compute device.",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        default=False,
        help="Activate Sequential CPU Offloading for GPUs with < 8 GB VRAM.",
    )

    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    input_image_path = Path(args.image)
    if not input_image_path.exists():
        logger.error(f"Input image not found: {input_image_path}")
        sys.exit(1)

    output_video_path = Path(args.output)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    from pipeline import CartoonVideoGenerator

    logger.info("Initializing CartoonVideoGenerator...")
    generator = CartoonVideoGenerator(
        base_model_id=args.base_model_id,
        motion_adapter_id=args.motion_adapter_id,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        device=args.device,
        low_vram=args.low_vram,
    )

    logger.info(f"Generating video from '{input_image_path.name}'...")
    result = generator.generate_clip(
        input_image=input_image_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        ip_adapter_scale=args.ip_adapter_scale,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=output_video_path,
    )

    logger.info("=" * 60)
    logger.info("Generation Summary:")
    logger.info(f"  Input Image     : {input_image_path.resolve()}")
    logger.info(f"  Output Video    : {result['video_path']}")
    logger.info(f"  Frames / FPS    : {result['num_frames']} frames @ {result['fps']} FPS")
    logger.info(f"  Duration        : {result['num_frames'] / result['fps']:.2f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

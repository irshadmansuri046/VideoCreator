#!/usr/bin/env python3
"""
Module 3: Memory-Efficient Image-to-Cartoon-Video Inference Engine
=================================================================
Part of an Open-Source Image-to-Cartoon-Video Generation Pipeline.

This module accepts a single still character image and synthesizes an animated
cartoon video clip (.mp4) using open-source, free-tier-compatible weights:
  1. Base Model: runwayml/stable-diffusion-v1-5
  2. Motion Engine: guoyww/animatediff-motion-adapter-v1-5-2 (AnimateDiff v2)
  3. Image Conditioning: IP-Adapter (h94/IP-Adapter with CLIP-ViT-H-14)
  4. Style Adapter: Custom cartoon LoRA weights from Module 2
  5. Scheduler: DDIMScheduler (linear beta schedule, linspace timesteps)

Engineered for 8GB–16GB VRAM GPUs (e.g. Google Colab / Kaggle T4):
  - VAE Slicing & Tiling to prevent memory spikes during high-res frame decoding.
  - Model CPU Offloading / Sequential CPU Offloading.
  - PyTorch 2.x SDPA memory-efficient attention.
  - Letterbox aspect-ratio preservation for non-square inputs.

Author: Senior Generative AI / Diffusion Pipeline Architect
License: Apache 2.0
"""

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CartoonVideoPipeline")


class CartoonVideoGenerator:
    """
    End-to-end Image-to-Cartoon-Video generation pipeline combining Stable Diffusion 1.5,
    AnimateDiff Motion Module v2, IP-Adapter identity preservation, and custom cartoon LoRA styling.
    """

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_weight_name: str = "ip-adapter_sd15.bin",
        image_encoder_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        lora_path: Optional[Union[str, Path]] = None,
        lora_scale: float = 0.85,
        device: str = "auto",
        low_vram: bool = False,
        enable_cpu_offload: bool = True,
        status_callback: Optional[Any] = None,
    ) -> None:
        self.base_model_id = base_model_id
        self.motion_adapter_id = motion_adapter_id
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_weight_name = ip_adapter_weight_name
        self.image_encoder_id = image_encoder_id
        self.lora_path = Path(lora_path) if lora_path else None
        self.lora_scale = float(lora_scale)
        self.low_vram = low_vram
        self.enable_cpu_offload = enable_cpu_offload
        self.device = self._resolve_device(device)
        self.status_callback = status_callback
        self.pipe = None

        self._initialize_pipeline()

    @staticmethod
    def check_cached_models() -> Dict[str, Any]:
        """
        Checks whether required foundation models are already downloaded
        in the local Hugging Face cache.
        """
        try:
            from huggingface_hub import try_to_load_from_cache
            status = {
                "motion_adapter": try_to_load_from_cache("guoyww/animatediff-motion-adapter-v1-5-2", "config.json") is not None,
                "base_model": (
                    try_to_load_from_cache("runwayml/stable-diffusion-v1-5", "unet/diffusion_pytorch_model.safetensors") is not None
                    or try_to_load_from_cache("runwayml/stable-diffusion-v1-5", "unet/diffusion_pytorch_model.bin") is not None
                ),
                "ip_adapter": try_to_load_from_cache("h94/IP-Adapter", "models/ip-adapter_sd15.bin") is not None,
            }
            status["all_cached"] = all(status.values())
            return status
        except Exception:
            return {"motion_adapter": False, "base_model": False, "ip_adapter": False, "all_cached": False}

    def _notify_status(self, progress_frac: float, message: str) -> None:
        """Helper to invoke status callback if registered."""
        if self.status_callback is not None:
            try:
                self.status_callback(progress_frac, message)
            except Exception:
                pass

    @staticmethod
    def _resolve_device(requested_device: str) -> object:
        """Determines best available compute backend (CUDA, Apple Silicon MPS, or CPU)."""
        req = requested_device.lower().strip()
        try:
            import torch
            if req == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            elif req == "cpu":
                return torch.device("cpu")
            elif req == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device("cpu")
        except ImportError:
            return "cpu"

    def _initialize_pipeline(self) -> None:
        """Loads and configures models with memory optimizations."""
        try:
            import torch
            from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
        except ImportError as e:
            logger.error("Diffusers and PyTorch are required: pip install diffusers torch")
            raise e

        device_type = getattr(self.device, "type", str(self.device)).lower()
        torch_dtype = torch.float16 if device_type in ("cuda", "mps") else torch.float32

        self._notify_status(0.05, f"Loading AnimateDiff motion adapter: '{self.motion_adapter_id}'...")
        logger.info(f"Loading AnimateDiff motion adapter: '{self.motion_adapter_id}' (device: {device_type}, dtype: {torch_dtype})...")

        try:
            motion_adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_id,
                torch_dtype=torch_dtype,
            )

            self._notify_status(0.15, f"Loading Stable Diffusion base pipeline: '{self.base_model_id}'...")
            logger.info(f"Initializing AnimateDiff pipeline with base model: '{self.base_model_id}'...")
            self.pipe = AnimateDiffPipeline.from_pretrained(
                self.base_model_id,
                motion_adapter=motion_adapter,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )

            # Configure DDIM Scheduler for smooth temporal consistency
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                beta_schedule="linear",
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1,
            )

            # Load IP-Adapter for strong image character conditioning
            self._notify_status(0.25, f"Loading IP-Adapter weights from '{self.ip_adapter_repo}'...")
            logger.info(f"Loading IP-Adapter weights from '{self.ip_adapter_repo}'...")
            try:
                self.pipe.load_ip_adapter(
                    self.ip_adapter_repo,
                    subfolder="models",
                    weight_name=self.ip_adapter_weight_name,
                    image_encoder_folder=self.image_encoder_id,
                )
                logger.info("IP-Adapter loaded successfully.")
            except Exception as ip_err:
                logger.warning(
                    f"Standard IP-Adapter loading with image_encoder_folder failed ({ip_err}). "
                    "Attempting fallback load without explicit subfolder..."
                )
                self.pipe.load_ip_adapter(
                    self.ip_adapter_repo,
                    subfolder="models",
                    weight_name=self.ip_adapter_weight_name,
                )

            # Load custom cartoon style LoRA if provided
            if self.lora_path is not None:
                self._notify_status(0.30, f"Loading custom cartoon LoRA from '{self.lora_path}'...")
                self._load_lora_weights()

            # Apply memory optimizations for 8GB–16GB VRAM
            self._notify_status(0.35, "Configuring memory optimizations (FP16, attention slicing, VAE slicing)...")
            self._apply_memory_optimizations()

            logger.info("CartoonVideoGenerator pipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize CartoonVideoGenerator: {e}")
            raise e

    def _load_lora_weights(self) -> None:
        """Loads and activates custom cartoon LoRA weights from Module 2."""
        if not self.lora_path.exists():
            logger.warning(f"LoRA path does not exist: {self.lora_path}. Proceeding without style LoRA.")
            return

        logger.info(f"Loading custom cartoon LoRA from: '{self.lora_path}' (scale: {self.lora_scale})...")
        try:
            if self.lora_path.is_file():
                self.pipe.load_lora_weights(
                    str(self.lora_path.parent),
                    weight_name=self.lora_path.name,
                    adapter_name="cartoon_lora",
                )
            else:
                self.pipe.load_lora_weights(
                    str(self.lora_path),
                    adapter_name="cartoon_lora",
                )

            # Set LoRA adapter scale
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(["cartoon_lora"], adapter_weights=[self.lora_scale])
            elif hasattr(self.pipe, "fuse_lora"):
                self.pipe.fuse_lora(lora_scale=self.lora_scale)
            logger.info("Custom cartoon LoRA weights successfully loaded and scaled.")
        except Exception as e:
            logger.warning(f"Error loading cartoon LoRA weights: {e}. Continuing with base pipeline.")

    def _apply_memory_optimizations(self) -> None:
        """Applies VAE slicing, tiling, attention slicing, and CPU offloading to prevent OOM."""
        try:
            import torch
        except ImportError:
            torch = None

        # VAE decoding optimization: slice and tile latents into sub-blocks
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
            logger.debug("VAE slicing enabled.")

        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
            logger.debug("VAE tiling enabled.")

        # Attention Slicing: drastically cuts activation RAM during temporal/spatial attention
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing(slice_size=1)
            logger.info("Attention slicing enabled (slice_size=1) for reduced memory footprint.")

        # PyTorch 2.x SDPA or xFormers attention
        try:
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory-efficient attention enabled.")
        except Exception:
            logger.debug("Using native PyTorch 2.x Scaled Dot Product Attention (SDPA).")

        # Device offloading logic
        device_type = getattr(self.device, "type", str(self.device)).lower()
        if device_type == "cuda" and torch is not None and torch.cuda.is_available():
            if self.low_vram:
                logger.info("Low VRAM mode: Enabling Sequential CPU Offloading (< 8 GB VRAM mode)...")
                self.pipe.enable_sequential_cpu_offload()
            elif self.enable_cpu_offload:
                logger.info("Enabling Model CPU Offloading (8 GB – 16 GB VRAM budget)...")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)
        elif device_type == "mps":
            logger.info("Apple Silicon MPS mode: Placing model pipeline in MPS memory...")
            if torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            self.pipe.to(self.device)
        else:
            self.pipe.to(self.device)

    @staticmethod
    def preprocess_image(
        input_image: Union[str, Path, object],
        target_size: int = 512,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> object:
        """
        Preprocesses arbitrary input images for the CLIP image encoder:
          - Converts color space cleanly to RGB.
          - Applies aspect-ratio-preserving letterbox scaling (no stretching or distortion).
          - Centers image on a target_size x target_size canvas.
        """
        try:
            from PIL import Image, ImageOps
        except ImportError as e:
            logger.error("Pillow is required for image preprocessing: pip install Pillow")
            raise e

        # Load image if file path is provided
        if isinstance(input_image, (str, Path)):
            img = Image.open(input_image)
            img.load()
        else:
            img = input_image

        # Respect EXIF camera orientation
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Handle transparency cleanly
        has_transparency = False
        if hasattr(img, "mode"):
            if img.mode in ("RGBA", "LA"):
                has_transparency = True
            elif img.mode == "P" and isinstance(getattr(img, "info", None), dict) and "transparency" in img.info:
                has_transparency = True

        if has_transparency:
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, bg_color)
            bg.paste(rgba, mask=rgba.split()[3])
            img = bg
        elif hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        if orig_w == target_size and orig_h == target_size:
            return img

        # Compute letterbox scale preserving aspect ratio
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        resample_filter = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
        resized_img = img.resize((new_w, new_h), resample=resample_filter)

        # Create centered canvas
        canvas = Image.new("RGB", (target_size, target_size), bg_color)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        canvas.paste(resized_img, (paste_x, paste_y))

        return canvas

    def generate_clip(
        self,
        input_image: Union[str, Path, object],
        prompt: str = "in cartoon artstyle, character talking, blinking, expressive smiling, waving",
        negative_prompt: str = "photorealistic, 3d, realistic skin, deformed, blurry, low quality, static, bad anatomy",
        ip_adapter_scale: float = 0.6,
        num_frames: int = 16,
        fps: int = 8,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        step_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generates an animated cartoon video clip conditioned on input_image and text prompt.

        Returns a dictionary containing:
          - 'video_path': path to generated .mp4 file
          - 'frames': list of raw PIL Image frames
          - 'fps': video framerate
          - 'num_frames': frame count
        """
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            torch = None

        if self.pipe is None:
            raise RuntimeError("Pipeline has not been initialized.")

        # 1. Preprocess input image for IP-Adapter
        processed_image = self.preprocess_image(input_image, target_size=512)

        # 2. Set IP-Adapter conditioning strength
        if hasattr(self.pipe, "set_ip_adapter_scale"):
            self.pipe.set_ip_adapter_scale(float(ip_adapter_scale))

        # 3. Configure random seed generator
        generator = None
        if seed is not None and has_torch:
            device_type = getattr(self.device, "type", str(self.device)).lower()
            gen_device = "mps" if device_type == "mps" else ("cuda" if device_type == "cuda" else "cpu")
            try:
                generator = torch.Generator(device=gen_device).manual_seed(int(seed))
            except Exception:
                generator = torch.Generator(device="cpu").manual_seed(int(seed))

        logger.info(f"Generating {num_frames} frames at {fps} FPS (Steps: {num_inference_steps}, CFG: {guidance_scale})...")
        logger.info(f"Motion Prompt: \"{prompt}\"")

        # 4. Prepare pipeline call arguments
        pipe_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "ip_adapter_image": processed_image,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if generator is not None:
            pipe_kwargs["generator"] = generator

        if step_callback is not None:
            def _step_end_callback(pipe_obj, step_idx, timestep, callback_kwargs):
                try:
                    step_callback(step_idx, num_inference_steps)
                except Exception:
                    pass
                return callback_kwargs

            pipe_kwargs["callback_on_step_end"] = _step_end_callback

        # 5. Run Diffusion Inference with OOM Guard
        try:
            try:
                if has_torch and hasattr(torch, "inference_mode"):
                    with torch.inference_mode():
                        output = self.pipe(**pipe_kwargs)
                else:
                    output = self.pipe(**pipe_kwargs)
            except TypeError as te:
                # Fallback if mock or older diffusers pipeline doesn't accept callback_on_step_end
                if "callback_on_step_end" in str(te):
                    pipe_kwargs.pop("callback_on_step_end", None)
                    if has_torch and hasattr(torch, "inference_mode"):
                        with torch.inference_mode():
                            output = self.pipe(**pipe_kwargs)
                    else:
                        output = self.pipe(**pipe_kwargs)
                else:
                    raise te

            frames = output.frames[0]  # List of PIL Images

        except Exception as err:
            err_msg = str(err).lower()
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None) if has_torch and hasattr(torch, "cuda") else None
            is_oom = "out of memory" in err_msg or (oom_type is not None and isinstance(err, oom_type))
            if is_oom:
                logger.error(
                    "Out-Of-Memory (OOM) error detected during video generation!\n"
                    "Actionable recovery steps:\n"
                    "  1. Reduce --num_frames (e.g. from 24 to 12 or 8).\n"
                    "  2. Initialize CartoonVideoGenerator with low_vram=True (Sequential CPU Offload).\n"
                    "  3. Reduce --num_inference_steps (e.g. to 15 steps).\n"
                    "  4. Ensure no other GPU processes are occupying VRAM / unified memory."
                )
                gc.collect()
                if has_torch:
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
            raise err

        # 5. Export to H.264 MP4 Video
        if output_path is None:
            output_path = Path("output_cartoon_clip.mp4")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_video(frames, output_path, fps=fps)

        logger.info(f"Successfully generated cartoon video clip: {output_path.resolve()} ({len(frames)} frames @ {fps} fps)")

        return {
            "video_path": str(output_path.resolve()),
            "frames": frames,
            "fps": fps,
            "num_frames": len(frames),
        }

    @staticmethod
    def export_video(frames: List[object], output_path: Union[str, Path], fps: int = 8) -> str:
        """
        Exports a list of PIL Images into high-compatibility H.264 .mp4 format.
        Tries diffusers.utils.export_to_video first, then imageio[ffmpeg], then OpenCV.
        """
        output_path = Path(output_path)

        # Strategy 1: diffusers built-in export_to_video
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, str(output_path), fps=fps)
            return str(output_path)
        except Exception as e1:
            logger.debug(f"diffusers.utils.export_to_video not used ({e1}); trying imageio...")

        # Strategy 2: imageio with ffmpeg
        try:
            import imageio
            import numpy as np

            np_frames = [np.array(f) for f in frames]
            imageio.mimsave(
                str(output_path),
                np_frames,
                fps=fps,
                codec="libx264",
                format="mp4",
            )
            return str(output_path)
        except Exception as e2:
            logger.debug(f"imageio failed ({e2}); trying cv2.VideoWriter...")

        # Strategy 3: OpenCV VideoWriter fallback
        try:
            import cv2
            import numpy as np

            first_frame = np.array(frames[0])
            h, w, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))

            for f in frames:
                bgr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            return str(output_path)
        except Exception as e3:
            logger.error(f"All video export strategies failed: {e3}")
            raise e3

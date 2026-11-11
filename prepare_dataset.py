#!/usr/bin/env python3
"""
Module 1: Image & Video Dataset Curation, Preprocessing, and Multimodal Auto-Captioning
======================================================================================
Part of an Open-Source Image-to-Cartoon-Video Generation Pipeline.

This module automates the curation and preparation of high-quality image-text paired
datasets for fine-tuning Stable Diffusion 1.5 LoRA models:
  1. Ingestion: Ingests still images and extracts keyframes from video files (1 FPS or scene changes).
  2. Preprocessing: Standardizes color space (RGB), strips alpha channels cleanly via compositing,
     and applies aspect-ratio-preserving center-crop with Lanczos resampling to 512x512.
  3. Auto-Captioning: Batched multimodal caption generation using lightweight Vision-Language Models
     (Salesforce/blip-image-captioning-base or microsoft/Florence-2-base) via Hugging Face Transformers.
  4. Packaging: Produces paired `image_XXXX.png` and `image_XXXX.txt` files with activation tokens.

Author: Senior AI Engineer / Python Developer
License: Apache 2.0
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

# Set up logging with timestamps and clean formatting
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetPreparer")

# Graceful tqdm import with generator fallback if missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        """Fallback lightweight iterator when tqdm is not yet installed."""
        return iterable

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}


@dataclass
class PreprocessingConfig:
    """Configuration parameters for dataset preparation pipeline."""
    input_dir: Path
    output_dir: Path
    trigger_word: str = "in cartoon artstyle,"
    target_size: int = 512
    fps: float = 1.0
    extraction_mode: str = "fps"  # "fps", "scene", or "both"
    scene_threshold: float = 0.35
    device: str = "auto"
    model_id: str = "Salesforce/blip-image-captioning-base"
    batch_size: int = 8
    max_new_tokens: int = 64
    prefix: str = "image"
    min_digits: int = 4
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    recursive: bool = True
    overwrite: bool = False
    skip_captioning: bool = False


class VideoKeyframeExtractor:
    """Extracts keyframes from video files at fixed FPS intervals or via scene transition detection."""

    def __init__(
        self,
        target_fps: float = 1.0,
        extraction_mode: str = "fps",
        scene_threshold: float = 0.35,
    ) -> None:
        self.target_fps = max(0.01, float(target_fps))
        self.extraction_mode = extraction_mode.lower()
        self.scene_threshold = float(scene_threshold)

    def extract_frames(self, video_path: Union[str, Path]) -> Generator:
        """
        Extract frames from a video file as RGB PIL Images.
        Yields (frame_idx, pil_image).
        """
        video_path = Path(video_path)
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as e:
            logger.error(
                "OpenCV and/or Pillow are required for video extraction. "
                "Please install them via: pip install opencv-python-headless Pillow"
            )
            raise e

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video file (corrupted or unsupported codec): {video_path}")
            return

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0 or np.isnan(native_fps):
            logger.warning(f"Invalid video FPS ({native_fps}) for {video_path.name}; defaulting to 24.0")
            native_fps = 24.0

        # Calculate frame step for fixed-rate sampling (e.g. 1 FPS)
        frame_interval = max(1, int(round(native_fps / self.target_fps)))
        min_scene_gap = max(1, int(native_fps * 0.25))  # Minimum 250ms gap between consecutive scene cuts

        current_frame_idx = 0
        extracted_count = 0
        prev_gray_frame = None
        last_extracted_idx = -min_scene_gap

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                is_fps_sample = (
                    self.extraction_mode in ("fps", "both")
                    and (current_frame_idx % frame_interval == 0)
                )

                is_scene_cut = False
                if self.extraction_mode in ("scene", "both") and self.scene_threshold > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray_frame is not None:
                        # Mean absolute frame difference normalized to [0, 1]
                        frame_diff = np.mean(
                            np.abs(gray.astype(np.float32) - prev_gray_frame.astype(np.float32))
                        ) / 255.0

                        if frame_diff >= self.scene_threshold:
                            # Avoid bursts within the same split-second transition
                            if (current_frame_idx - last_extracted_idx) >= min_scene_gap:
                                is_scene_cut = True
                    prev_gray_frame = gray

                if is_fps_sample or is_scene_cut:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        yield (extracted_count, pil_img)
                        extracted_count += 1
                        last_extracted_idx = current_frame_idx
                    except Exception as err:
                        logger.warning(f"Error converting frame {current_frame_idx} in {video_path.name}: {err}")

                current_frame_idx += 1
        finally:
            cap.release()


class ImagePreprocessor:
    """
    Standardizes images for Stable Diffusion 1.5 LoRA training:
      - Strips alpha channels cleanly by compositing over solid background.
      - Normalizes color spaces (RGBA, LA, CMYK, Palette, Grayscale) to clean RGB.
      - Center-crops to 1:1 square aspect ratio without stretching or distortion.
      - Resamples to exact target dimensions (512x512) using Lanczos interpolation.
    """

    def __init__(self, target_size: int = 512, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        self.target_size = target_size
        self.bg_color = bg_color

    def preprocess(self, img_input: Union[str, Path, object]) -> object:
        """
        Loads and standardizes an image.
        Returns a processed PIL Image (RGB, target_size x target_size).
        """
        from PIL import Image, ImageOps

        # Load image if path provided
        if isinstance(img_input, (str, Path)):
            img = Image.open(img_input)
            img.load()  # Force load to catch truncated/corrupted files early
        else:
            img = img_input

        # Respect EXIF orientation tags (e.g. from camera/smartphone stills)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # 1. Clean color space conversion to RGB (handling alpha cleanly)
        img = self.normalize_to_rgb(img, self.bg_color)

        # 2. Aspect-ratio preserving center crop to square (1:1)
        img = self.center_crop_square(img)

        # 3. High quality Lanczos resampling to target size
        resample_filter = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
        if hasattr(img, "size") and img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), resample=resample_filter)

        return img

    @staticmethod
    def normalize_to_rgb(img: object, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> object:
        """
        Converts any PIL image mode to clean RGB.
        Composites RGBA/LA/transparent palette images over a solid canvas to prevent black-halo artifacts.
        """
        try:
            from PIL import Image
            has_pil = True
        except ImportError:
            has_pil = False

        if not hasattr(img, "mode"):
            return img

        # Transparent palette or alpha-channel modes
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and hasattr(img, "info") and "transparency" in img.info):
            if has_pil:
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, bg_color)
                alpha_mask = rgba.split()[3]
                background.paste(rgba, mask=alpha_mask)
                return background
            elif hasattr(img, "convert"):
                return img.convert("RGB")
        elif img.mode != "RGB":
            if hasattr(img, "convert"):
                return img.convert("RGB")
        return img

    @staticmethod
    def center_crop_square(img: object) -> object:
        """Crops the image to a centered square box matching its shortest dimension."""
        if not hasattr(img, "size"):
            return img

        width, height = img.size
        if width == height:
            return img

        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim

        return img.crop((left, top, right, bottom))


class AutoCaptioner:
    """
    Multimodal Auto-Captioning using lightweight Vision-Language Models
    (Salesforce/blip-image-captioning-base or microsoft/Florence-2-base) via Hugging Face Transformers.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-base",
        trigger_word: str = "in cartoon artstyle,",
        device: str = "auto",
        batch_size: int = 8,
        max_new_tokens: int = 64,
    ) -> None:
        self.model_id = model_id
        self.trigger_word = trigger_word
        self.batch_size = max(1, batch_size)
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)
        self.model = None
        self.processor = None
        self._is_florence = "florence" in model_id.lower()

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
            else:
                logger.warning(f"Device '{requested_device}' not available/recognized. Falling back to CPU.")
                return torch.device("cpu")
        except ImportError:
            # Fallback when torch is not yet installed in testing/verification environments
            logger.debug("PyTorch not installed. Falling back to CPU.")
            return "cpu"

    def load_model(self) -> None:
        """Loads model weights and processor into memory on target device."""
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            logger.error("Hugging Face Transformers and PyTorch are required for caption generation.")
            raise e

        logger.info(f"Loading captioning model '{self.model_id}' on device '{self.device}'...")

        # Select model architecture
        if self._is_florence:
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
        else:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            model_class = BlipForConditionalGeneration

        device_type = getattr(self.device, "type", str(self.device))
        torch_dtype = torch.float16 if device_type == "cuda" else torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = model_class.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            logger.info("VLM captioning model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vision-language model '{self.model_id}': {e}")
            raise e

    def format_caption(self, raw_caption: str) -> str:
        """
        Cleans and prepends trigger activation token to raw caption.
        Example:
          raw_caption = "a young anime girl smiling with cherry blossoms in the background"
          trigger_word = "in cartoon artstyle,"
          result = "in cartoon artstyle, a young anime girl smiling with cherry blossoms in the background"
        """
        clean_raw = raw_caption.strip()
        clean_trigger = self.trigger_word.strip()

        if not clean_trigger:
            return clean_raw

        # Format trigger ending cleanly
        if not clean_trigger.endswith((",", ";", ".")):
            clean_trigger = f"{clean_trigger},"

        # Avoid redundant duplication if model echoed part of trigger
        if clean_raw.lower().startswith(clean_trigger.lower()):
            return clean_raw

        return f"{clean_trigger} {clean_raw}".strip()

    def generate_captions_batch(self, images: List[object]) -> List[str]:
        """
        Generates captions for a batch of PIL Images using the loaded VLM.
        """
        try:
            import torch
        except ImportError as e:
            logger.error("PyTorch is required for generating captions.")
            raise e

        if self.model is None or self.processor is None:
            self.load_model()

        if not images:
            return []

        results = []
        with torch.inference_mode():
            if self._is_florence:
                # Florence-2 task prompt
                prompt = "<DETAILED_CAPTION>"
                for img in images:
                    inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=self.max_new_tokens,
                        num_beams=3,
                    )
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = self.processor.post_process_generation(
                        generated_text, task=prompt, image_size=(img.width, img.height)
                    )
                    caption_text = parsed_answer.get(prompt, generated_text)
                    results.append(self.format_caption(caption_text))
            else:
                # BLIP Architecture (batch capable)
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=10,
                )
                decoded_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                for cap in decoded_captions:
                    results.append(self.format_caption(cap))

        return results


class DatasetPipeline:
    """Coordinates file discovery, extraction, preprocessing, batch captioning, and dataset export."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.extractor = VideoKeyframeExtractor(
            target_fps=config.fps,
            extraction_mode=config.extraction_mode,
            scene_threshold=config.scene_threshold,
        )
        self.preprocessor = ImagePreprocessor(
            target_size=config.target_size,
            bg_color=config.bg_color,
        )
        self.captioner = None
        if not config.skip_captioning:
            self.captioner = AutoCaptioner(
                model_id=config.model_id,
                trigger_word=config.trigger_word,
                device=config.device,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
            )

    def scan_inputs(self) -> Tuple[List[Path], List[Path]]:
        """Scans input directory for supported still images and video files."""
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.config.input_dir}")

        image_files = []
        video_files = []

        pattern = "**/*" if self.config.recursive else "*"
        for path in self.config.input_dir.glob(pattern):
            if path.is_file():
                ext = path.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.append(path)
                elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                    video_files.append(path)

        image_files.sort()
        video_files.sort()
        return image_files, video_files

    def run(self) -> dict:
        """Executes the full dataset preparation pipeline."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        image_files, video_files = self.scan_inputs()

        logger.info(
            f"Discovered {len(image_files)} still image(s) and {len(video_files)} video file(s) "
            f"in '{self.config.input_dir}'"
        )

        if not image_files and not video_files:
            logger.warning("No supported image or video files found. Exiting.")
            return {"processed_images": 0, "corrupted_files": 0, "pairs_saved": 0}

        # Estimate total items to dynamically calculate zero-padding width
        estimated_total = len(image_files) + len(video_files) * 30
        pad_width = max(self.config.min_digits, len(str(estimated_total)))

        processed_images = []  # List of tuples: (save_path_png, save_path_txt, pil_image)
        corrupted_count = 0
        current_id = 1

        # -------------------------------------------------------------
        # Step 1: Ingestion & Preprocessing
        # -------------------------------------------------------------
        logger.info("--- Step 1/3: Ingesting and Preprocessing Images & Video Frames ---")
        
        # 1a: Process still images
        for img_path in tqdm(image_files, desc="Preprocessing images", unit="img"):
            try:
                processed_pil = self.preprocessor.preprocess(img_path)
                file_stem = f"{self.config.prefix}_{current_id:0{pad_width}d}"
                out_png = self.config.output_dir / f"{file_stem}.png"
                out_txt = self.config.output_dir / f"{file_stem}.txt"

                if out_png.exists() and not self.config.overwrite:
                    logger.debug(f"Skipping existing file: {out_png.name}")
                else:
                    if hasattr(processed_pil, "save"):
                        processed_pil.save(out_png, format="PNG", optimize=True)

                processed_images.append((out_png, out_txt, processed_pil))
                current_id += 1
            except Exception as e:
                logger.warning(f"Skipping corrupted image '{img_path.name}': {e}")
                corrupted_count += 1

        # 1b: Extract and process video keyframes
        for vid_path in tqdm(video_files, desc="Extracting video keyframes", unit="vid"):
            try:
                for frame_idx, raw_pil_frame in self.extractor.extract_frames(vid_path):
                    try:
                        processed_pil = self.preprocessor.preprocess(raw_pil_frame)
                        file_stem = f"{self.config.prefix}_{current_id:0{pad_width}d}"
                        out_png = self.config.output_dir / f"{file_stem}.png"
                        out_txt = self.config.output_dir / f"{file_stem}.txt"

                        if not out_png.exists() or self.config.overwrite:
                            if hasattr(processed_pil, "save"):
                                processed_pil.save(out_png, format="PNG", optimize=True)

                        processed_images.append((out_png, out_txt, processed_pil))
                        current_id += 1
                    except Exception as fe:
                        logger.warning(f"Frame {frame_idx} processing error in '{vid_path.name}': {fe}")
                        corrupted_count += 1
            except Exception as ve:
                logger.warning(f"Skipping corrupted video '{vid_path.name}': {ve}")
                corrupted_count += 1

        logger.info(f"Successfully preprocessed {len(processed_images)} standard 512x512 images.")

        # -------------------------------------------------------------
        # Step 2: Batch Multimodal Auto-Captioning
        # -------------------------------------------------------------
        pairs_saved = 0
        if not self.config.skip_captioning and processed_images:
            logger.info("--- Step 2/3: Generating Captions with Multimodal VLM ---")
            
            # Filter items that already have captions if overwrite is False
            pending_items = [
                item for item in processed_images
                if not item[1].exists() or self.config.overwrite
            ]

            batch_size = self.config.batch_size
            total_batches = (len(pending_items) + batch_size - 1) // batch_size

            for i in tqdm(range(0, len(pending_items), batch_size), desc="Captioning batches", total=total_batches):
                batch = pending_items[i : i + batch_size]
                batch_images = [item[2] for item in batch]
                batch_txt_paths = [item[1] for item in batch]

                try:
                    captions = self.captioner.generate_captions_batch(batch_images)
                    for txt_path, caption in zip(batch_txt_paths, captions):
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(caption + "\n")
                        pairs_saved += 1
                except Exception as ce:
                    logger.error(f"Batch caption generation failed: {ce}. Falling back to default trigger phrase.")
                    for txt_path in batch_txt_paths:
                        fallback_caption = self.config.trigger_word.strip()
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(fallback_caption + "\n")
                        pairs_saved += 1
        elif self.config.skip_captioning:
            logger.info("Captioning skipped as requested (--skip_captioning).")
            # Write trigger word fallback caption file
            for _, txt_path, _ in processed_images:
                if not txt_path.exists() or self.config.overwrite:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(self.config.trigger_word.strip() + "\n")
                    pairs_saved += 1
        else:
            pairs_saved = len(processed_images)

        # -------------------------------------------------------------
        # Step 3: Summary & Verification
        # -------------------------------------------------------------
        logger.info("--- Step 3/3: Pipeline Execution Summary ---")
        summary = {
            "input_images_found": len(image_files),
            "input_videos_found": len(video_files),
            "preprocessed_images": len(processed_images),
            "corrupted_skipped": corrupted_count,
            "training_pairs_ready": len(list(self.config.output_dir.glob(f"{self.config.prefix}_*.png"))),
            "target_resolution": f"{self.config.target_size}x{self.config.target_size}",
            "output_directory": str(self.config.output_dir.resolve()),
        }

        logger.info("=" * 60)
        for k, v in summary.items():
            logger.info(f"  {k:25s}: {v}")
        logger.info("=" * 60)

        return summary


def parse_args(args: Optional[List[str]] = None) -> PreprocessingConfig:
    """Parses and validates CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Module 1: Preprocess cartoon/anime images & videos and generate paired LoRA training captions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input directory containing raw images (.png, .jpg) or videos (.mp4, .mov).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for 512x512 standardized images and paired .txt caption files.",
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default="in cartoon artstyle,",
        help="Activation token / style trigger phrase prepended to every generated caption.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for vision-language model inference.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="Hugging Face model ID for captioning (e.g., Salesforce/blip-image-captioning-base or microsoft/Florence-2-base).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size for vision-language model caption generation.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target square resolution (pixels) for SD 1.5 training.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame extraction rate (frames per second) for video inputs.",
    )
    parser.add_argument(
        "--extraction_mode",
        type=str,
        default="fps",
        choices=["fps", "scene", "both"],
        help="Video frame extraction strategy: fixed FPS ('fps'), scene changes ('scene'), or combination ('both').",
    )
    parser.add_argument(
        "--scene_threshold",
        type=float,
        default=0.35,
        help="Threshold (0.0 to 1.0) for detecting inter-frame scene cuts when extraction_mode is 'scene' or 'both'.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated token length for VLM captions.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="image",
        help="Base filename prefix for output files (e.g. 'image' -> 'image_0001.png').",
    )
    parser.add_argument(
        "--min_digits",
        type=int,
        default=4,
        help="Minimum zero-padding width for output filenames.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively traverse nested subdirectories in input_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files instead of skipping them.",
    )
    parser.add_argument(
        "--skip_captioning",
        action="store_true",
        default=False,
        help="Only run video extraction and image preprocessing without loading the VLM captioner.",
    )

    parsed = parser.parse_args(args)

    return PreprocessingConfig(
        input_dir=Path(parsed.input_dir),
        output_dir=Path(parsed.output_dir),
        trigger_word=parsed.trigger_word,
        target_size=parsed.target_size,
        fps=parsed.fps,
        extraction_mode=parsed.extraction_mode,
        scene_threshold=parsed.scene_threshold,
        device=parsed.device,
        model_id=parsed.model_id,
        batch_size=parsed.batch_size,
        max_new_tokens=parsed.max_new_tokens,
        prefix=parsed.prefix,
        min_digits=parsed.min_digits,
        recursive=parsed.recursive,
        overwrite=parsed.overwrite,
        skip_captioning=parsed.skip_captioning,
    )


def main() -> None:
    """Entry point for CLI execution."""
    config = parse_args()
    pipeline = DatasetPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

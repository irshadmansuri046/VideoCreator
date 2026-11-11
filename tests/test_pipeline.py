#!/usr/bin/env python3
"""
Unit Tests for Module 3: Image-to-Cartoon-Video Pipeline
========================================================
Tests aspect-ratio scaling (letterboxing), CLI argument parsing,
pipeline initialization, memory offloading configurations, and OOM error handling.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure workspace root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_test import parse_args
from pipeline import CartoonVideoGenerator


class TestImagePreprocessing(unittest.TestCase):
    """Test aspect-ratio preserving letterbox scaling for non-square inputs."""

    def test_letterbox_widescreen(self):
        # 16:9 input (1920x1080) -> should scale to 512x288 and pad top/bottom to 512x512
        mock_pil = MagicMock()
        mock_pil.ImageOps.exif_transpose.side_effect = lambda x: x
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_img.size = (1920, 1080)
        mock_img.info = {}

        mock_resized = MagicMock()
        mock_img.resize.return_value = mock_resized

        mock_canvas = MagicMock()
        mock_pil.Image.new.return_value = mock_canvas

        with patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_pil.Image, "PIL.ImageOps": mock_pil.ImageOps}):
            res = CartoonVideoGenerator.preprocess_image(mock_img, target_size=512)
            mock_img.resize.assert_called_once()
            args, kwargs = mock_img.resize.call_args
            self.assertEqual(args[0], (512, 288))
            mock_pil.Image.new.assert_called_with("RGB", (512, 512), (255, 255, 255))
            mock_canvas.paste.assert_called_with(mock_resized, (0, 112))

    def test_letterbox_portrait(self):
        # 9:16 input (1080x1920) -> should scale to 288x512 and pad left/right to 512x512
        mock_pil = MagicMock()
        mock_pil.ImageOps.exif_transpose.side_effect = lambda x: x
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_img.size = (1080, 1920)
        mock_img.info = {}

        mock_resized = MagicMock()
        mock_img.resize.return_value = mock_resized
        mock_canvas = MagicMock()
        mock_pil.Image.new.return_value = mock_canvas

        with patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_pil.Image, "PIL.ImageOps": mock_pil.ImageOps}):
            res = CartoonVideoGenerator.preprocess_image(mock_img, target_size=512)
            args, kwargs = mock_img.resize.call_args
            self.assertEqual(args[0], (288, 512))
            mock_canvas.paste.assert_called_with(mock_resized, (112, 0))

    def test_letterbox_square_passthrough(self):
        # Exact 512x512 input -> returns original without resizing
        mock_pil = MagicMock()
        mock_pil.ImageOps.exif_transpose.side_effect = lambda x: x
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_img.size = (512, 512)
        mock_img.info = {}

        with patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_pil.Image, "PIL.ImageOps": mock_pil.ImageOps}):
            res = CartoonVideoGenerator.preprocess_image(mock_img, target_size=512)
            mock_img.resize.assert_not_called()
            self.assertEqual(res, mock_img)


class TestCLIOptions(unittest.TestCase):
    """Test CLI argument parsing for cli_test.py."""

    def test_cli_defaults(self):
        args = parse_args(["--image", "character.png"])
        self.assertEqual(args.image, "character.png")
        self.assertEqual(args.output, "output.mp4")
        self.assertEqual(args.num_frames, 16)
        self.assertEqual(args.fps, 8)
        self.assertEqual(args.ip_adapter_scale, 0.6)
        self.assertEqual(args.num_inference_steps, 25)
        self.assertEqual(args.guidance_scale, 7.5)
        self.assertEqual(args.seed, 42)
        self.assertFalse(args.low_vram)

    def test_cli_custom_overrides(self):
        args = parse_args([
            "--image", "anime.jpg",
            "--prompt", "walking in the rain",
            "--output", "clip_final.mp4",
            "--lora_path", "weights/cartoon.safetensors",
            "--lora_scale", "0.9",
            "--ip_adapter_scale", "0.75",
            "--num_frames", "24",
            "--fps", "12",
            "--num_inference_steps", "30",
            "--low_vram",
        ])
        self.assertEqual(args.prompt, "walking in the rain")
        self.assertEqual(args.output, "clip_final.mp4")
        self.assertEqual(args.lora_scale, 0.9)
        self.assertEqual(args.ip_adapter_scale, 0.75)
        self.assertEqual(args.num_frames, 24)
        self.assertEqual(args.fps, 12)
        self.assertEqual(args.num_inference_steps, 30)
        self.assertTrue(args.low_vram)


class TestGeneratorLifecycle(unittest.TestCase):
    """Test pipeline initialization, memory offloading configurations, and OOM handling."""

    @patch.object(CartoonVideoGenerator, "_initialize_pipeline")
    def test_device_resolution(self, mock_init):
        gen = CartoonVideoGenerator(device="cpu")
        self.assertEqual(str(gen.device), "cpu")

    @patch.object(CartoonVideoGenerator, "_initialize_pipeline")
    def test_low_vram_offload_selection(self, mock_init):
        gen = CartoonVideoGenerator(low_vram=True)
        self.assertTrue(gen.low_vram)

    @patch.object(CartoonVideoGenerator, "_initialize_pipeline")
    def test_oom_recovery_handling(self, mock_init):
        gen = CartoonVideoGenerator()
        gen.pipe = MagicMock()
        # Simulate OutOfMemoryError
        gen.pipe.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        with patch.object(CartoonVideoGenerator, "preprocess_image", return_value=MagicMock()):
            with self.assertRaises(RuntimeError) as ctx:
                gen.generate_clip(input_image=MagicMock(), prompt="test")
            self.assertIn("out of memory", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

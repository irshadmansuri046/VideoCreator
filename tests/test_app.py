#!/usr/bin/env python3
"""
Unit Tests for Module 4: Gradio Web Application
===============================================
Tests CLI configuration, animated GIF conversion, lazy singleton generator caching,
and UI event wiring.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure workspace root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (
    _GENERATOR_INSTANCE,
    build_app,
    export_animated_gif,
    get_generator,
    parse_args,
)


class TestAppArgumentParsing(unittest.TestCase):
    """Test CLI argument parsing for app.py."""

    def test_default_args(self):
        args = parse_args([])
        self.assertEqual(args.server_name, "0.0.0.0")
        self.assertEqual(args.server_port, 7860)
        self.assertFalse(args.share)
        self.assertIsNone(args.lora_path)
        self.assertEqual(args.lora_scale, 0.85)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.low_vram)

    def test_custom_args(self):
        args = parse_args([
            "--server_name", "127.0.0.1",
            "--server_port", "8080",
            "--share",
            "--lora_path", "./models/cartoon.safetensors",
            "--lora_scale", "0.95",
            "--device", "cuda",
            "--low_vram",
        ])
        self.assertEqual(args.server_name, "127.0.0.1")
        self.assertEqual(args.server_port, 8080)
        self.assertTrue(args.share)
        self.assertEqual(args.lora_path, "./models/cartoon.safetensors")
        self.assertEqual(args.lora_scale, 0.95)
        self.assertEqual(args.device, "cuda")
        self.assertTrue(args.low_vram)


class TestGIFExport(unittest.TestCase):
    """Test animated GIF creation from frame sequence."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_animated_gif(self):
        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()
        gif_path = self.out_dir / "test_anim.gif"

        res = export_animated_gif([mock_frame1, mock_frame2], gif_path, fps=8)
        self.assertEqual(res, str(gif_path.resolve()))
        mock_frame1.save.assert_called_once()
        args, kwargs = mock_frame1.save.call_args
        self.assertEqual(args[0], gif_path)
        self.assertTrue(kwargs.get("save_all"))
        self.assertEqual(kwargs.get("append_images"), [mock_frame2])
        # 1000 / 8 = 125ms duration
        self.assertEqual(kwargs.get("duration"), 125)


class TestSingletonGenerator(unittest.TestCase):
    """Test lazy singleton caching of the CartoonVideoGenerator."""

    def test_get_generator_singleton(self):
        import app

        # Reset global singleton for test
        app._GENERATOR_INSTANCE = None

        with patch("pipeline.CartoonVideoGenerator") as mock_gen_class:
            mock_instance = MagicMock()
            mock_gen_class.return_value = mock_instance

            # First call: should instantiate
            gen1 = get_generator(lora_path="test_lora.safetensors")
            self.assertEqual(gen1, mock_instance)
            mock_gen_class.assert_called_once()

            # Second call: should reuse cached instance
            gen2 = get_generator(lora_path="test_lora.safetensors")
            self.assertEqual(gen2, mock_instance)
            self.assertEqual(mock_gen_class.call_count, 1)

        # Cleanup
        app._GENERATOR_INSTANCE = None


class TestGradioAppConstruction(unittest.TestCase):
    """Test build_app wiring with mocked Gradio."""

    def test_build_app_structure(self):
        mock_gr = MagicMock()
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks

        with patch.dict("sys.modules", {"gradio": mock_gr}):
            demo = build_app(lora_path="lora.safetensors")
            mock_gr.Blocks.assert_called_once()
            mock_gr.Image.assert_called_once()
            mock_gr.Video.assert_called_once()
            mock_gr.Button.assert_called_once()

    def test_build_app_mps_adaptive_defaults(self):
        mock_gr = MagicMock()
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks

        with patch("pipeline.CartoonVideoGenerator._resolve_device", return_value=MagicMock(type="mps")):
            with patch.dict("sys.modules", {"gradio": mock_gr}):
                demo = build_app(device="mps")
                # Ensure Slider was configured
                slider_calls = [c for c in mock_gr.Slider.call_args_list]
                self.assertTrue(len(slider_calls) >= 2)

    def test_build_app_cuda_adaptive_defaults(self):
        mock_gr = MagicMock()
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks

        with patch("pipeline.CartoonVideoGenerator._resolve_device", return_value=MagicMock(type="cuda")):
            with patch.dict("sys.modules", {"gradio": mock_gr}):
                demo = build_app(device="cuda")
                slider_calls = [c for c in mock_gr.Slider.call_args_list]
                self.assertTrue(len(slider_calls) >= 2)


if __name__ == "__main__":
    unittest.main()

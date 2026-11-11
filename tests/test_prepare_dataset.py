#!/usr/bin/env python3
"""
Unit and Integration Tests for Module 1: Dataset Preparation Pipeline
=====================================================================
Tests CLI argument parsing, cropping mathematics, color space normalization,
caption formatting, video extraction logic, batching, and error handling.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import (
    AutoCaptioner,
    DatasetPipeline,
    ImagePreprocessor,
    PreprocessingConfig,
    VideoKeyframeExtractor,
    parse_args,
)


class TestArgumentParsing(unittest.TestCase):
    """Test CLI argument parsing and config defaults."""

    def test_default_arguments(self):
        config = parse_args(["--input_dir", "/tmp/inputs", "--output_dir", "/tmp/outputs"])
        self.assertEqual(config.input_dir, Path("/tmp/inputs"))
        self.assertEqual(config.output_dir, Path("/tmp/outputs"))
        self.assertEqual(config.trigger_word, "in cartoon artstyle,")
        self.assertEqual(config.target_size, 512)
        self.assertEqual(config.fps, 1.0)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.model_id, "Salesforce/blip-image-captioning-base")
        self.assertFalse(config.skip_captioning)
        self.assertEqual(config.min_digits, 4)

    def test_custom_arguments(self):
        config = parse_args([
            "--input_dir", "raw_data",
            "--output_dir", "lora_dataset",
            "--trigger_word", "sks anime style,",
            "--target_size", "768",
            "--fps", "2.5",
            "--device", "cpu",
            "--batch_size", "16",
            "--skip_captioning",
            "--overwrite",
        ])
        self.assertEqual(config.trigger_word, "sks anime style,")
        self.assertEqual(config.target_size, 768)
        self.assertEqual(config.fps, 2.5)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 16)
        self.assertTrue(config.skip_captioning)
        self.assertTrue(config.overwrite)


class TestCaptionFormatting(unittest.TestCase):
    """Test activation token / trigger phrase formatting."""

    def test_trigger_formatting_with_comma(self):
        captioner = AutoCaptioner(trigger_word="in cartoon artstyle,")
        raw = "a boy with blue hair smiling"
        formatted = captioner.format_caption(raw)
        self.assertEqual(formatted, "in cartoon artstyle, a boy with blue hair smiling")

    def test_trigger_formatting_without_comma(self):
        captioner = AutoCaptioner(trigger_word="in cartoon artstyle")
        raw = "a scenic mountain view"
        formatted = captioner.format_caption(raw)
        self.assertEqual(formatted, "in cartoon artstyle, a scenic mountain view")

    def test_trigger_formatting_no_duplication(self):
        captioner = AutoCaptioner(trigger_word="in cartoon artstyle,")
        raw = "in cartoon artstyle, a cat running"
        formatted = captioner.format_caption(raw)
        self.assertEqual(formatted, "in cartoon artstyle, a cat running")

    def test_empty_trigger_word(self):
        captioner = AutoCaptioner(trigger_word="")
        raw = "a futuristic robot city"
        formatted = captioner.format_caption(raw)
        self.assertEqual(formatted, "a futuristic robot city")


class TestImagePreprocessing(unittest.TestCase):
    """Test aspect-ratio preserving center cropping and color space conversion."""

    def test_center_crop_landscape(self):
        mock_img = MagicMock()
        mock_img.size = (1920, 1080)
        # Expected crop: width 1920, height 1080 -> square 1080x1080 from center
        # left = (1920 - 1080) // 2 = 420
        # top = (1080 - 1080) // 2 = 0
        # right = 420 + 1080 = 1500
        # bottom = 1080
        ImagePreprocessor.center_crop_square(mock_img)
        mock_img.crop.assert_called_once_with((420, 0, 1500, 1080))

    def test_center_crop_portrait(self):
        mock_img = MagicMock()
        mock_img.size = (1080, 1920)
        # Expected crop: width 1080, height 1920 -> square 1080x1080 from center
        # left = 0
        # top = (1920 - 1080) // 2 = 420
        # right = 1080
        # bottom = 420 + 1080 = 1500
        ImagePreprocessor.center_crop_square(mock_img)
        mock_img.crop.assert_called_once_with((0, 420, 1080, 1500))

    def test_center_crop_already_square(self):
        mock_img = MagicMock()
        mock_img.size = (512, 512)
        res = ImagePreprocessor.center_crop_square(mock_img)
        self.assertEqual(res, mock_img)
        mock_img.crop.assert_not_called()

    def test_normalize_rgb_clean(self):
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        res = ImagePreprocessor.normalize_to_rgb(mock_img)
        self.assertEqual(res, mock_img)
        mock_img.convert.assert_not_called()

    def test_normalize_cmyk_to_rgb(self):
        mock_img = MagicMock()
        mock_img.mode = "CMYK"
        mock_converted = MagicMock()
        mock_img.convert.return_value = mock_converted
        res = ImagePreprocessor.normalize_to_rgb(mock_img)
        mock_img.convert.assert_called_once_with("RGB")
        self.assertEqual(res, mock_converted)


class TestVideoKeyframeExtractor(unittest.TestCase):
    """Test video frame sampling rate and scene transition logic."""

    def test_frame_interval_calculation(self):
        extractor = VideoKeyframeExtractor(target_fps=1.0)
        self.assertEqual(extractor.target_fps, 1.0)
        self.assertEqual(extractor.extraction_mode, "fps")

        # 30 native fps / 1 target fps -> interval 30
        native_fps = 30.0
        interval = max(1, int(round(native_fps / extractor.target_fps)))
        self.assertEqual(interval, 30)

        # 24 native fps / 2 target fps -> interval 12
        extractor_2fps = VideoKeyframeExtractor(target_fps=2.0)
        interval_2fps = max(1, int(round(24.0 / extractor_2fps.target_fps)))
        self.assertEqual(interval_2fps, 12)


class TestPipelineOrchestration(unittest.TestCase):
    """Test pipeline input scanning, batching, and file pair generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "inputs"
        self.output_dir = Path(self.temp_dir.name) / "outputs"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_input_scanning(self):
        # Create dummy image and video files
        (self.input_dir / "test1.png").touch()
        (self.input_dir / "test2.jpg").touch()
        (self.input_dir / "clip.mp4").touch()
        (self.input_dir / "readme.txt").touch()  # Unsupported, should be ignored

        config = PreprocessingConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            skip_captioning=True,
        )
        pipeline = DatasetPipeline(config)
        images, videos = pipeline.scan_inputs()

        self.assertEqual(len(images), 2)
        self.assertEqual(len(videos), 1)
        self.assertIn(self.input_dir / "test1.png", images)
        self.assertIn(self.input_dir / "test2.jpg", images)
        self.assertIn(self.input_dir / "clip.mp4", videos)

    @patch("prepare_dataset.tqdm", side_effect=lambda x, *args, **kwargs: x)
    def test_pipeline_skip_captioning_run(self, mock_tqdm):
        # Create 2 synthetic test images
        img1 = self.input_dir / "sample_a.png"
        img2 = self.input_dir / "sample_b.jpg"
        img1.touch()
        img2.touch()

        config = PreprocessingConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            trigger_word="in cartoon artstyle,",
            skip_captioning=True,
            min_digits=4,
        )
        pipeline = DatasetPipeline(config)

        # Mock preprocessor to write real PNG dummy files when save() is called
        def mock_save(path, *args, **kwargs):
            Path(path).touch()

        mock_pil = MagicMock()
        mock_pil.save.side_effect = mock_save
        pipeline.preprocessor.preprocess = MagicMock(return_value=mock_pil)

        summary = pipeline.run()

        self.assertEqual(summary["input_images_found"], 2)
        self.assertEqual(summary["preprocessed_images"], 2)
        self.assertEqual(summary["corrupted_skipped"], 0)
        self.assertEqual(summary["training_pairs_ready"], 2)

        # Verify that output png and txt files with trigger word were created
        png1 = self.output_dir / "image_0001.png"
        txt1 = self.output_dir / "image_0001.txt"
        png2 = self.output_dir / "image_0002.png"
        txt2 = self.output_dir / "image_0002.txt"

        self.assertTrue(png1.exists())
        self.assertTrue(txt1.exists())
        self.assertTrue(png2.exists())
        self.assertTrue(txt2.exists())

        with open(txt1, "r", encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "in cartoon artstyle,")

    @patch("prepare_dataset.tqdm", side_effect=lambda x, *args, **kwargs: x)
    def test_pipeline_with_mocked_captioner(self, mock_tqdm):
        # Test full pipeline with mocked batch captioner
        img1 = self.input_dir / "anime_scene.png"
        img1.touch()

        config = PreprocessingConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            trigger_word="in cartoon artstyle,",
            skip_captioning=False,
            min_digits=4,
        )
        pipeline = DatasetPipeline(config)

        # Mock preprocessor save
        mock_pil = MagicMock()
        mock_pil.save.side_effect = lambda p, *args, **kwargs: Path(p).touch()
        pipeline.preprocessor.preprocess = MagicMock(return_value=mock_pil)

        # Mock captioner generate_captions_batch
        pipeline.captioner = MagicMock()
        pipeline.captioner.generate_captions_batch.return_value = [
            "in cartoon artstyle, a warrior holding a glowing sword in the forest"
        ]

        summary = pipeline.run()
        self.assertEqual(summary["preprocessed_images"], 1)

        txt1 = self.output_dir / "image_0001.txt"
        self.assertTrue(txt1.exists())
        with open(txt1, "r", encoding="utf-8") as f:
            content = f.read().strip()
            self.assertEqual(
                content,
                "in cartoon artstyle, a warrior holding a glowing sword in the forest"
            )

    @patch("prepare_dataset.tqdm", side_effect=lambda x, *args, **kwargs: x)
    def test_corrupted_file_handling(self, mock_tqdm):
        # Create a file that fails preprocessing
        bad_img = self.input_dir / "corrupted.png"
        bad_img.touch()

        config = PreprocessingConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            skip_captioning=True,
        )
        pipeline = DatasetPipeline(config)

        # Simulate exception during preprocessing
        pipeline.preprocessor.preprocess = MagicMock(side_effect=IOError("Corrupt JPEG data"))

        summary = pipeline.run()
        self.assertEqual(summary["corrupted_skipped"], 1)
        self.assertEqual(summary["preprocessed_images"], 0)


if __name__ == "__main__":
    unittest.main()

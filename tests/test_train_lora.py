#!/usr/bin/env python3
"""
Unit Tests for Module 2: Stable Diffusion 1.5 LoRA Training Engine
==================================================================
Tests dataset ingestion, paired caption resolution, collate functions,
optimizer fallback logic, and CLI configuration.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure workspace root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_lora import (
    CartoonLoRADataset,
    collate_fn,
    parse_args,
    save_lora_checkpoint,
    select_optimizer,
)


class TestLoRAArgumentParsing(unittest.TestCase):
    """Test CLI arguments and default configurations for LoRA training."""

    def test_default_cli_args(self):
        args = parse_args(["--dataset_dir", "/tmp/dataset", "--output_dir", "/tmp/lora_out"])
        self.assertEqual(args.dataset_dir, "/tmp/dataset")
        self.assertEqual(args.output_dir, "/tmp/lora_out")
        self.assertEqual(args.pretrained_model_name_or_path, "runwayml/stable-diffusion-v1-5")
        self.assertEqual(args.trigger_word, "in cartoon artstyle,")
        self.assertEqual(args.lora_rank, 16)
        self.assertEqual(args.lora_alpha, 16)
        self.assertEqual(args.train_batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.mixed_precision, "fp16")
        self.assertTrue(args.use_8bit_adam)
        self.assertTrue(args.gradient_checkpointing)
        self.assertFalse(args.train_text_encoder)

    def test_custom_cli_args(self):
        args = parse_args([
            "--dataset_dir", "data/curated",
            "--output_dir", "models/lora_v1",
            "--lora_rank", "32",
            "--lora_alpha", "64",
            "--train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "5e-5",
            "--train_text_encoder",
            "--no_8bit_adam",
            "--mixed_precision", "bf16",
        ])
        self.assertEqual(args.lora_rank, 32)
        self.assertEqual(args.lora_alpha, 64)
        self.assertEqual(args.train_batch_size, 2)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertTrue(args.train_text_encoder)
        self.assertFalse(args.use_8bit_adam)
        self.assertEqual(args.mixed_precision, "bf16")


class TestCartoonLoRADataset(unittest.TestCase):
    """Test dataset scanning, paired caption resolution, and missing caption fallback."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_scanning_with_captions(self):
        img1 = self.dataset_dir / "image_0001.png"
        txt1 = self.dataset_dir / "image_0001.txt"
        img2 = self.dataset_dir / "image_0002.jpg"
        txt2 = self.dataset_dir / "image_0002.txt"

        img1.touch()
        txt1.write_text("in cartoon artstyle, a smiling knight", encoding="utf-8")
        img2.touch()
        txt2.write_text("in cartoon artstyle, a magical dragon", encoding="utf-8")

        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 77

        dataset = CartoonLoRADataset(
            dataset_dir=self.dataset_dir,
            tokenizer=mock_tokenizer,
            trigger_word="in cartoon artstyle,",
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.samples[0][0], img1)
        self.assertEqual(dataset.samples[0][1], txt1)

    def test_dataset_missing_caption_fallback(self):
        img1 = self.dataset_dir / "image_0001.png"
        img1.touch()

        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 77
        mock_tokenizer.return_value = MagicMock(input_ids=MagicMock(squeeze=lambda dim: [1, 2, 3]))

        mock_pil = MagicMock()
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_pil.Image.open.return_value = mock_img

        with patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_pil.Image}):
            dataset = CartoonLoRADataset(
                dataset_dir=self.dataset_dir,
                tokenizer=mock_tokenizer,
                trigger_word="in cartoon artstyle,",
            )
            item = dataset[0]
            # Caption should fall back to trigger word
            self.assertEqual(item["caption"], "in cartoon artstyle,")


class TestCollateFunction(unittest.TestCase):
    """Test collate_fn tensor batching."""

    def test_collate_fn_mock_tensors(self):
        mock_torch = MagicMock()
        mock_tensor1 = MagicMock()
        mock_tensor2 = MagicMock()
        mock_ids1 = MagicMock()
        mock_ids2 = MagicMock()

        batch = [
            {"pixel_values": mock_tensor1, "input_ids": mock_ids1, "caption": "cap1"},
            {"pixel_values": mock_tensor2, "input_ids": mock_ids2, "caption": "cap2"},
        ]

        with patch.dict("sys.modules", {"torch": mock_torch}):
            mock_torch.stack.side_effect = lambda x: MagicMock(to=lambda memory_format: MagicMock(float=lambda: "stacked_pixels")) if x[0] == mock_tensor1 else "stacked_ids"
            res = collate_fn(batch)
            self.assertIn("pixel_values", res)
            self.assertIn("input_ids", res)
            self.assertEqual(res["captions"], ["cap1", "cap2"])


class TestOptimizerSelection(unittest.TestCase):
    """Test 8-bit AdamW vs standard AdamW fallback logic."""

    def test_fallback_to_standard_adamw(self):
        mock_param = MagicMock()
        mock_torch = MagicMock()
        mock_adamw = MagicMock()
        mock_torch.optim.AdamW.return_value = mock_adamw

        # Mock bitsandbytes being unavailable
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("train_lora.logger"):
                opt, name = select_optimizer([mock_param], learning_rate=1e-4, use_8bit_adam=True)
                self.assertIn("Standard 32-bit AdamW", name)

    def test_explicit_standard_adamw(self):
        mock_param = MagicMock()
        mock_torch = MagicMock()
        mock_adamw = MagicMock()
        mock_torch.optim.AdamW.return_value = mock_adamw

        with patch.dict("sys.modules", {"torch": mock_torch}):
            opt, name = select_optimizer([mock_param], learning_rate=1e-4, use_8bit_adam=False)
            self.assertIn("Standard 32-bit AdamW", name)


class TestCheckpointSaving(unittest.TestCase):
    """Test saving weights and metadata in .safetensors format."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_lora_checkpoint_metadata(self):
        mock_unet = MagicMock()
        mock_accelerator = MagicMock()
        mock_accelerator.unwrap_model.return_value = mock_unet

        metadata = {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "lora_rank": 16,
            "lora_alpha": 16,
            "trigger_word": "in cartoon artstyle,",
            "step": 500,
        }

        save_lora_checkpoint(
            output_dir=self.output_dir,
            unet=mock_unet,
            text_encoder=None,
            accelerator=mock_accelerator,
            metadata=metadata,
            is_final=False,
            step=500,
        )

        chkpt_dir = self.output_dir / "checkpoint-500"
        self.assertTrue(chkpt_dir.exists())
        metadata_file = chkpt_dir / "lora_metadata.json"
        self.assertTrue(metadata_file.exists())

        with open(metadata_file, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
            self.assertEqual(saved_meta["lora_rank"], 16)
            self.assertEqual(saved_meta["step"], 500)

        mock_unet.save_pretrained.assert_called_once_with(chkpt_dir, safe_serialization=True)


if __name__ == "__main__":
    unittest.main()

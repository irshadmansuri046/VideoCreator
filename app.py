#!/usr/bin/env python3
"""
Module 4: Production Gradio Web Application for Image-to-Cartoon-Video
=====================================================================
Part of an Open-Source Image-to-Cartoon-Video Generation Pipeline.

This web application exposes the Module 3 `CartoonVideoGenerator` via a responsive,
modern Gradio UI (Blocks v4+):
  - Drag-and-drop character image input with live preview.
  - Action/motion prompt box with collapsible negative prompt accordion.
  - Advanced motion sliders (IP-Adapter strength, frame count, FPS, CFG, seed).
  - Autoplaying and looping video player for rendered .mp4 clips.
  - Animated .gif export toggle and instant download.
  - Real-time progress updates with step indicators.
  - Graceful CUDA Out-Of-Memory (OOM) handling with user feedback banners.
  - One-click preset examples for instant testing.
  - CLI configuration with optional public tunneling (--share).

Author: Full-Stack Generative AI / UI Engineer
License: Apache 2.0
"""

import argparse
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CartoonVideoApp")

# Global singleton pipeline instance
_GENERATOR_INSTANCE = None


def get_generator(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    lora_path: Optional[Union[str, Path]] = None,
    lora_scale: float = 0.85,
    device: str = "auto",
    low_vram: bool = False,
) -> object:
    """
    Returns the cached CartoonVideoGenerator singleton, initializing it lazily
    on first call to optimize startup time and avoid duplicate VRAM allocations.
    """
    global _GENERATOR_INSTANCE
    if _GENERATOR_INSTANCE is None:
        from pipeline import CartoonVideoGenerator

        logger.info("Initializing global CartoonVideoGenerator singleton...")
        _GENERATOR_INSTANCE = CartoonVideoGenerator(
            base_model_id=base_model_id,
            motion_adapter_id=motion_adapter_id,
            lora_path=lora_path,
            lora_scale=lora_scale,
            device=device,
            low_vram=low_vram,
            enable_cpu_offload=True,
        )
    return _GENERATOR_INSTANCE


def export_animated_gif(frames: List[object], output_path: Union[str, Path], fps: int = 8) -> str:
    """Exports a sequence of PIL images into an optimized animated .gif file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = max(20, int(1000 / max(1, fps)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return str(output_path.resolve())


def build_app(
    lora_path: Optional[str] = None,
    lora_scale: float = 0.85,
    device: str = "auto",
    low_vram: bool = False,
) -> object:
    """Constructs and returns the Gradio Blocks web application."""
    try:
        import gradio as gr
    except ImportError as e:
        logger.error("Gradio is required to run the web app. Install via: pip install gradio>=4.20.0")
        raise e

    # Read custom CSS
    css_path = Path(__file__).parent / "custom.css"
    custom_css = ""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()

    # Hugging Face ZeroGPU compatibility decorator
    try:
        import spaces
        gpu_decorator = spaces.GPU(duration=75)
        logger.info("Hugging Face ZeroGPU detected: GPU decorator active.")
    except (ImportError, Exception):
        def gpu_decorator(fn):
            return fn

    # Pre-warm or configure generator settings
    @gpu_decorator
    def generate_video_handler(
        input_image: Optional[object],
        prompt: str,
        negative_prompt: str,
        ip_adapter_scale: float,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        randomize_seed: bool,
        export_gif: bool,
        progress=gr.Progress(track_tqdm=True),
    ) -> Tuple[Optional[str], Optional[str], str]:
        """Handles video generation requests with progress tracking and OOM safety."""
        if input_image is None:
            raise gr.Error("Please upload or drag-and-drop a character portrait image before generating.")

        if not prompt or not prompt.strip():
            prompt = "in cartoon artstyle, character talking and smiling, expressive animation, waving hand"

        # Resolve random seed
        actual_seed = random.randint(0, 2147483647) if randomize_seed else int(seed)

        start_time = time.time()
        progress(0.1, desc="Initializing generator and preprocessing image...")

        try:
            generator = get_generator(
                lora_path=lora_path,
                lora_scale=lora_scale,
                device=device,
                low_vram=low_vram,
            )

            progress(0.3, desc=f"Synthesizing {num_frames} cartoon frames with AnimateDiff & IP-Adapter...")
            output_dir = Path(tempfile.gettempdir()) / "cartoon_video_app"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            video_filename = output_dir / f"cartoon_anim_{timestamp}.mp4"

            # Execute generation
            result = generator.generate_clip(
                input_image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_scale=float(ip_adapter_scale),
                num_frames=int(num_frames),
                fps=int(fps),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                seed=actual_seed,
                output_path=video_filename,
            )

            progress(0.85, desc="Encoding output formats...")
            gif_path = None
            if export_gif and result.get("frames"):
                gif_filename = output_dir / f"cartoon_anim_{timestamp}.gif"
                gif_path = export_animated_gif(result["frames"], gif_filename, fps=int(fps))

            elapsed = time.time() - start_time
            progress(1.0, desc="Video generated successfully!")

            # Format summary card
            metadata_md = f"""
### 📊 Generation Summary
- **Render Time:** `{elapsed:.2f}s` ({len(result['frames'])/elapsed:.2f} fps generation speed)
- **Video Specs:** `{result['num_frames']} frames` @ `{result['fps']} FPS` (~`{result['num_frames']/result['fps']:.1f}s` duration)
- **Effective Seed:** `{actual_seed}`
- **IP-Adapter Weight:** `{ip_adapter_scale:.2f}` | **CFG Scale:** `{guidance_scale:.1f}`
            """.strip()

            return str(video_filename), gif_path, metadata_md

        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda" in err_msg:
                warning_msg = (
                    "⚠️ GPU Memory Limit Reached (CUDA OOM)!\n\n"
                    "Suggestions to recover:\n"
                    "1. Lower **Number of Frames** to 16.\n"
                    "2. Lower **Inference Steps** to 20.\n"
                    "3. Relaunch server with `--low_vram` flag for Sequential CPU Offloading."
                )
                logger.error(warning_msg)
                raise gr.Error(warning_msg)
            else:
                logger.error(f"Generation error: {e}")
                raise gr.Error(f"Error during video synthesis: {e}")

    # Build Gradio Blocks UI
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=custom_css, title="CartoonVideoAI - Image to Animated Video") as demo:
        # Header Section
        gr.HTML("""
        <div class="app-header">
            <h1>🎬 Image-to-Cartoon-Video AI Studio</h1>
            <p>
                Transform any character portrait into an animated cartoon video clip with consistent motion, 
                facial preservation, and fluid anime cel-shading.
                <span class="badge-tag">AnimateDiff v2 + IP-Adapter + SD 1.5 LoRA</span>
            </p>
        </div>
        """)

        with gr.Row():
            # Left Column: Inputs & Model Controls
            with gr.Column(scale=5):
                gr.Markdown("### 1. Character Image Input")
                input_image = gr.Image(
                    type="pil",
                    label="Drop Character Image (PNG / JPG / WebP)",
                    sources=["upload", "clipboard"],
                    height=280,
                )

                gr.Markdown("### 2. Motion Prompt & Action")
                prompt = gr.Textbox(
                    lines=3,
                    label="Animation Prompt",
                    placeholder="e.g. in cartoon artstyle, character laughing and turning head, expressive eyes, dynamic wind blowing hair",
                    value="in cartoon artstyle, character talking and smiling, expressive animation, waving hand",
                )

                with gr.Accordion("Negative Prompt", open=False):
                    negative_prompt = gr.Textbox(
                        lines=2,
                        label="Negative Tags",
                        value="photorealistic, 3d, realistic skin, deformed, blurry, low quality, static, bad anatomy, bad eyes",
                    )

                with gr.Accordion("⚙️ Advanced Motion & Generation Settings", open=False):
                    ip_adapter_scale = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="IP-Adapter Identity Fidelity",
                        info="Controls how closely the animated character adheres to the source image face/clothing.",
                    )

                    with gr.Row():
                        num_frames = gr.Slider(
                            minimum=16,
                            maximum=32,
                            value=16,
                            step=8,
                            label="Frame Count",
                            info="16 frames (~2 sec) or 24 frames (~3 sec).",
                        )
                        fps = gr.Slider(
                            minimum=6,
                            maximum=16,
                            value=8,
                            step=1,
                            label="Framerate (FPS)",
                            info="Video playback speed.",
                        )

                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=15,
                            maximum=40,
                            value=25,
                            step=1,
                            label="Inference Steps",
                            info="Higher steps increase detail at cost of generation time.",
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale (CFG)",
                            info="Controls alignment with prompt.",
                        )

                    with gr.Row():
                        seed = gr.Number(
                            value=42,
                            label="Random Seed",
                            precision=0,
                        )
                        randomize_seed = gr.Checkbox(
                            value=True,
                            label="Randomize Seed",
                        )

                    export_gif = gr.Checkbox(
                        value=True,
                        label="Also Export Animated .GIF (for Discord/Slack sharing)",
                    )

                generate_btn = gr.Button(
                    "✨ Generate Cartoon Video",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            # Right Column: Output Video & Downloadables
            with gr.Column(scale=5):
                gr.Markdown("### 3. Animated Video Player")
                video_output = gr.Video(
                    label="Rendered Cartoon Video (.mp4)",
                    autoplay=True,
                    loop=True,
                    height=360,
                    elem_classes=["video-preview"],
                )

                gif_output = gr.File(
                    label="Download Animated GIF (.gif)",
                    visible=True,
                )

                metadata_output = gr.Markdown(
                    value="*Generation metrics and model diagnostics will appear here once rendering is complete.*",
                    elem_classes=["metadata-card"],
                )

        # Preset Examples Section
        gr.Markdown("### 🌟 Instant Preset Examples (Click to Load)")
        example_dir = Path(__file__).parent / "assets" / "examples"
        examples_data = []

        ex1 = example_dir / "example_portrait.png"
        ex2 = example_dir / "example_pet.png"
        ex3 = example_dir / "example_warrior.png"

        if ex1.exists():
            examples_data.append([
                str(ex1),
                "in cartoon artstyle, anime girl blinking, cheerful smiling, wind blowing hair, fluid cel-shading",
                0.6,
                16,
                8,
                25,
                7.5,
            ])
        if ex2.exists():
            examples_data.append([
                str(ex2),
                "in cartoon artstyle, cute pet tilting head, blinking curious eyes, dynamic tail wagging",
                0.65,
                16,
                8,
                25,
                7.5,
            ])
        if ex3.exists():
            examples_data.append([
                str(ex3),
                "in cartoon artstyle, fierce warrior readying stance, glowing magical aura, battle ready",
                0.7,
                16,
                8,
                25,
                7.5,
            ])

        if examples_data:
            gr.Examples(
                examples=examples_data,
                inputs=[
                    input_image,
                    prompt,
                    ip_adapter_scale,
                    num_frames,
                    fps,
                    num_inference_steps,
                    guidance_scale,
                ],
                label="Click any preset below to auto-fill inputs:",
            )

        # Wire click event
        generate_btn.click(
            fn=generate_video_handler,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                ip_adapter_scale,
                num_frames,
                fps,
                num_inference_steps,
                guidance_scale,
                seed,
                randomize_seed,
                export_gif,
            ],
            outputs=[
                video_output,
                gif_output,
                metadata_output,
            ],
        )

    return demo


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses CLI flags for server configuration and sharing."""
    parser = argparse.ArgumentParser(
        description="Module 4: Run the Image-to-Cartoon-Video Gradio Web Application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="IP address to bind the server to.",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Port number to host the server on.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio shareable link via temporary tunneling.",
    )
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
        help="Scale factor for the custom cartoon LoRA adapter.",
    )
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
        help="Activate Sequential CPU Offloading for systems with < 8 GB VRAM.",
    )

    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    logger.info(f"Launching Gradio Web App on {args.server_name}:{args.server_port} (share={args.share})...")

    demo = build_app(
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        device=args.device,
        low_vram=args.low_vram,
    )

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

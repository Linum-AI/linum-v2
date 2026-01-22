<p align="center">
  <a href="https://www.linum.ai/"><img src="assets/linum_logo.png" alt="Linum AI" width="200"></a>
</p>

<p align="center">
  <a href="https://www.linum.ai/field-notes/launch-linum-v2"><strong><font size="5">✨ Check out the launch</font></strong></a>
  <br><br>
  <a href="https://replicate.com/linum-ai/linum-v2-720p"><img src="assets/replicate.svg" height="32" alt="Replicate"> <strong><font size="5">Try on Replicate</font></strong></a>
</p>


# Linum v2: Text-to-Video Generation
Linum v2 is a pair of 2B parameter text-to-video generation models (360p or 720p, 2-5 seconds, 24 FPS).


## Installation

### Prerequisites

- Python 3.10-3.12
- NVIDIA GPU with CUDA 12.8 support

### Install with uv

First, install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install dependencies:

```bash
git clone https://github.com/Linum-AI/linum-v2.git
cd linum-v2
uv sync
```

### Flash Attention 3 (Optional, Hopper GPUs only)

For best performance on H100/H200 GPUs, install Flash Attention 3:

```bash
deactivate # deactivate any current env
hash -r  # clear shell's command cache
source .venv/bin/activate # reactivate
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install triton==3.5.0
```

If you see this warning don't worry, we need triton 3.5.0 in order to play with the bitsandbytes version used by diffusers:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.                                         
torch 2.7.1+cu128 requires triton==3.3.1; platform_system == "Linux", but you have triton 3.5.0 which is incompatible.
```

**Note:** Flash Attention 3 only works on Hopper architecture (H100, H200).
On other GPUs (e.g., Ada/RTX 4090), the code automatically falls back to
PyTorch's scaled_dot_product_attention (SDPA).

## Quick Start

Generate your first video:

```bash
# 720p (default)
uv run python generate_video.py \
    --prompt "In a charming hand-drawn 2D animation style, a rust-orange fox with cream chest fur and alert triangular ears grips a cherry-red steering wheel with both paws, its bushy tail curled on the passenger seat. Stylized trees and pastel houses whoosh past the windows in smooth parallax layers. The fox's golden eyes focus intently ahead, whiskers twitching as it navigates a winding country road rendered in soft watercolor textures." \
    --output fox.mp4 \
    --seed 20 \
    --cfg 7.0
```

https://github.com/user-attachments/assets/13c50d93-8a09-4aea-b12f-8695685d9665

```bash
# 360p (faster, lower VRAM)
uv run python generate_video.py \
    --prompt "A cute 3D animated baby goat with shaggy gray fur, a fluffy white chin tuft, and stubby curved horns perches on a round wooden stool. Warm golden studio lights bounce off its glossy cherry-red acoustic guitar as it rhythmically strums with a confident hoof, hind legs dangling. Framed family portraits of other barnyard animals line the cream-colored walls, a leafy potted ficus sits in the back corner, and dust motes drift through the cozy, sun-speckled room." \
    --output goat.mp4 \
    --seed 16 \
    --cfg 10.0 \
    --resolution 360p
```


https://github.com/user-attachments/assets/e37c9944-5fb4-4d85-80ff-fd6bc87e84d5

Weights are automatically downloaded from HuggingFace Hub on first run (~20GB per model).

## Usage

### Basic Usage

```bash
# 720p video, 2 seconds (default)
uv run python generate_video.py --prompt "Your prompt here" --output output.mp4

# 360p video, 2 seconds (faster, lower VRAM)
uv run python generate_video.py --prompt "Your prompt here" --output output.mp4 --resolution 360p

# 720p video, longer duration
uv run python generate_video.py --prompt "Your prompt here" --duration 4.0
```

### All Options

```bash
uv run python generate_video.py \
    --prompt "Your detailed prompt" \
    --output output.mp4 \
    --resolution 720p \
    --duration 2.0 \
    --seed 42 \
    --cfg 7.0 \
    --num_steps 50 \
    --negative_prompt "blurry, low quality"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (required) | Text description of the video |
| `--output` | `output.mp4` | Output file path |
| `--resolution` | `720p` | Resolution: `360p` or `720p` |
| `--duration` | `2.0` | Video duration in seconds (2.0-5.0) |
| `--seed` | `20` | Random seed for reproducibility |
| `--cfg` | `10.0` | Classifier-free guidance scale (recommended: 7-10; higher values follow prompts more closely but may oversaturate) |
| `--num_steps` | `50` | Number of sampler steps |
| `--negative_prompt` | `""` | What to avoid in generation |

### Using Local Weights

If you've downloaded weights manually:

```bash
uv run python generate_video.py \
    --prompt "Your prompt" \
    --model-path /path/to/dit.safetensors \
    --vae-path /path/to/vae.safetensors \
    --t5-encoder-path /path/to/t5/text_encoder \
    --t5-tokenizer-path /path/to/t5/tokenizer
```

## Hardware Requirements

| Resolution | VRAM Required |
|------------|---------------|
| 360p | ~25GB |
| 720p | ~35GB |

**Recommended GPUs:** H100, A100-80GB, or similar high-VRAM GPUs

### Speed Benchmarks (H100, 50 steps)

| Resolution | Duration | Generation Time |
|------------|----------|-----------------|
| 360p | 2 seconds | ~40 seconds |
| 360p | 5 seconds | ~2 minutes |
| 720p | 2 seconds | ~4 minutes |
| 720p | 5 seconds | ~15 minutes |

## Model Architecture

Linum V2 uses a Diffusion Transformer (DiT) architecture with:

- **DiT Backbone**: 2B parameters, trained from scratch with flow matching objective
- **Text Encoder**: T5-XXL
- **VAE**: WAN 2.1 VAE

## Model Weights

Weights are hosted on HuggingFace Hub:

- [Linum-AI/linum-v2-360p](https://huggingface.co/Linum-AI/linum-v2-360p) - 360p model
- [Linum-AI/linum-v2-720p](https://huggingface.co/Linum-AI/linum-v2-720p) - 720p model

## Citation

```bibtex
@software{linum_v2_2026,
  title = {Linum V2: Text-to-Video Generation},
  author = {Linum AI},
  year = {2026},
  url = {https://github.com/Linum-AI/linum-v2}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file.

## About Linum

Linum is a team of two brothers building a tiny-yet-powerful AI research lab. We train our own generative media models from scratch.

**Subscribe to [Field Notes](https://buttondown.com/linum-ai)** — technical deep dives on building generative video models from the ground up, plus updates on new releases from Linum.

**Contact:** hello@linum.ai — Reach out if you're selling high-quality video data.

## Acknowledgments

This project uses the following components under the Apache 2.0 License:

- [Wan Video 2.1 3D Causal Video VAE ](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py)
- [Google T5-XXL](https://github.com/google-research/text-to-text-transfer-transformer)
- [PyTorch](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers), [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)

Thank you to our investors and infrastructure partners:

<p align="center">
  <a href="https://www.ycombinator.com/"><img src="assets/yc.svg" alt="Y Combinator" height="36" style="vertical-align: middle"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://adverb.vc/"><img src="assets/adverb.svg" alt="Adverb Ventures" height="40" style="vertical-align: middle"></a>
</p>
<br><br>
<p align="center">
  <a href="https://crusoe.ai/"><img src="assets/crusoe.svg" alt="Crusoe" height="28" style="vertical-align: middle"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.together.ai/"><img src="assets/together.svg" alt="Together AI" height="28" style="vertical-align: middle"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.cloudflare.com/"><img src="assets/cloudflare.svg" alt="Cloudflare" height="32" style="vertical-align: middle"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.ubicloud.com/"><img src="assets/ubicloud.png" alt="Ubicloud" height="34" style="vertical-align: middle"></a>
</p>

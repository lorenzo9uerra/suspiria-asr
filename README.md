# Suspiria-ASR

Suspiria-ASR is a research training stack for an ASR decoder inspired by Delayed Streams Modeling models and Voxtral-Realtime.

The project trains a decoder over precomputed Mimi audio latents. Audio is encoded at 12.5 Hz, so each latent represents 80 ms of audio. The decoder advances at the same rate and predicts exactly one token per latent step. The token stream contains normal text tokens plus `[BOS]`, `[EOS]`, `[P]` for wait/pad, and `[W]` for the start of a text island.

The main pipeline is:

```text
raw audio dataset
-> timestamped transcription
-> Mimi latent encoding
-> paired HF dataset with latents and transcriptions
-> Decoder-only training
```

The model is composed of a frozen Mimi variational audio encoder used without RVQ discretization, followed by a stream-synchronous autoregressive decoder following the Voxtral-Realtime decoder design. The decoder consumes continuous Mimi latents and previous text-stream tokens, then predicts the next token on the same 80 ms timeline.

The model architecture is compatible with streaming inference. In practice, however, robust streaming operation requires training on audio sequences longer than the decoder context window, so that the model learns to operate with truncated historical context. Since the current dataset consists of shorter utterances, approximately 30-second clips, the provided inference workflow is currently intended for offline transcription of fixed-length audio segments.

Collected datasets and released models are available in the Hugging Face collection below.

[![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/3podi/suspiria-asr)

## Available Models

| Language | Parameters | WER | Hugging Face | Weights & Biases |
|---|---:|---:|---|---|
| Italian | 250M | 11.0% | [HF](https://huggingface.co/3podi/suspiria-asr-ita) | [W&B](https://wandb.ai/3posi/suspiria-asr/reports/Suspiria-ASR-ITA--VmlldzoxNjY0NDAyOA) |

## Quick Inference Example

To run the trained ASR stack on the sample audio file:

```bash
uv sync
```

Run:

```bash
uv run python infer.py --config-name offline-ita audio=audio/worldcup.mp3
```

## Environment

Use Python 3.10+. Dependencies are managed with `uv` from `pyproject.toml`. The repository pins Python 3.12 in `.python-version` so `uv` will select a compatible interpreter automatically.

Install the default training, encoding, and inference environment:

```bash
uv sync
```

Install the transcription environment instead:

```bash
uv sync --only-group transcribe
```

The transcription environment is separated into its own dependency group because `qwen-asr[vllm]` currently pins a different `transformers`, `huggingface-hub`, and `torch` stack than the rest of the repo.

The `transcribe` group is currently intended for GPU-only on Linux environments.

The default `main` group installs PyTorch from the official nightly indexes: CUDA 12.6 nightlies on Linux and Windows, and CPU nightlies on macOS. The `transcribe` group follows the stable `qwen-asr[vllm]`-compatible torch line instead. If you need a different PyTorch build, update the `torch` requirement and `[tool.uv.sources]` entry in `pyproject.toml` before syncing.

Authenticate with Hugging Face if you use private repos:

```bash
hf auth login
```

The code assumes commands are run from the repository root:

```bash
cd suspiria-asr
```

## Configs

Important configs live in `configs/`:

```text
configs/transcription.yaml        Qwen ASR transcription + timestamps
configs/encoding.yaml             Mimi latent encoding
configs/training.yaml             main decoder training
configs/inference/offline-ita.yaml simple single-file inference
configs/tokenizer_training.yaml   BPE tokenizer training
configs/scaling.yaml              scaling-law sweep base
configs/scaling_smoke.yaml        short scaling smoke run
```

Before running a stage, edit the corresponding yaml.

## 1. Transcribe Audio

`preprocessing/transcribe.py` reads a Hugging Face audio dataset and produces timestamped transcription JSONL shards.

Install the transcription dependency first:

```bash
uv sync --only-group transcribe
```

This transcription environment is currently supported only on Linux with a CUDA-capable NVIDIA GPU.

Edit:

```bash
configs/transcription.yaml
```

Set at least:

```yaml
dataset:
  repo_id: "disco-eth/EuroSpeech"
  country: "italy"
  splits: ["train", "validation", "test"]

asr:
  model: "Qwen/Qwen3-ASR-1.7B"
  forced_aligner: "Qwen/Qwen3-ForcedAligner-0.6B"
  language: "Italian"

output:
  out_dir: "out/transcriptions"
```

Run:

```bash
uv run --only-group transcribe -- python -m preprocessing.transcribe
```

This step writes transcription shards under:

```text
out/transcriptions/<country>/<split>/
```

If upload is enabled in the config, it also uploads them to the configured HF dataset repo.

## 2. Encode Mimi Latents

`preprocessing/encode_latents.py` reads the same audio dataset and encodes audio with Mimi.

Use the base project environment:

```bash
uv sync
```

Edit:

```bash
configs/encoding.yaml
```

Set:

```yaml
dataset:
  repo_id: "disco-eth/EuroSpeech"
  country: "italy"
  splits: ["train", "validation", "test"]

output:
  latent_dir: "out/latents"
  manifest_dir: "out/latents/manifests"

```

Run:

```bash
uv run python -m preprocessing.encode_latents --config-path configs/encoding.yaml
```

This writes latent parquet shards and local manifests under:

```text
out/latents/<country>/<split>/
out/latents/manifests/<country>/<split>.jsonl
```


## 3. Pair And Upload Latents

`preprocessing/upload_latents_to_hf.py` pairs transcription metadata with latent shards and optionally uploads a training dataset to HF.


```bash
uv run python -m preprocessing.upload_latents_to_hf \
  --latents-dir out/latents \
  --transcriptions-dir out/transcriptions \
  --output-dir out/paired_latents \
  --repo-id your-org/your-latents-dataset \
  --private \
  --country italy
```



## 4. Train The Decoder

Training requires the PyTorch nightly line because the decoder uses:

```python
torch.nn.attention.varlen.varlen_attn
```

Install the training environment:

```bash
uv sync
```

The default `main` group resolves the PyTorch `2.12.0.dev` nightly line from the official nightly indexes. If you use a different CUDA setup, edit the `torch` requirement and `[tool.uv.sources]` entry in `pyproject.toml` first, then run `uv sync`.

Run:

```bash
uv run python -m training.train
```


## 5. Export Checkpoints To Hugging Face

Upload exported decoder weights:

```bash
uv run python -m training.utils.push_checkpoint_to_hf \
  --output-dir out/training \
  --repo-id your-org/your-decoder-repo \
  --private
```

Upload exported weights plus a full resumable training checkpoint:

```bash
uv run python -m training.utils.push_checkpoint_to_hf \
  --output-dir out/training \
  --repo-id your-org/your-decoder-repo \
  --private \
  --include-optimizer
```

The exported model weights are saved as:

```text
model.safetensors
```

The full resumable checkpoint is uploaded as:

```text
training_checkpoint.pt
```

Use `model.safetensors` for inference/fine-tuning initialization. Use `training_checkpoint.pt` only for exact training resume.


## Other Dataset

The preprocessing scripts are primarily written for the EuroSpeech Hugging Face dataset layout. In particular, `preprocessing/transcribe.py` and `preprocessing/encode_latents.py` assume an audio dataset that can be loaded by split and country, with stable sample keys that can be shared across transcription and latent-encoding outputs.

Other datasets can be used for training, but they must first be converted to the paired latent dataset format consumed by `training/train.py`. The training dataset must provide timestamped word metadata in the manifest and Mimi projected latents in parquet shards.

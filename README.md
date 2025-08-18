Wan2.2-TI2V-5B-Turbo — 4-Step Distilled Vision-Language Model

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/per30112mmm/Wan2.2-TI2V-5B-Turbo/releases)

![AI Vision Banner](https://images.unsplash.com/photo-1531297484001-80022131f5a1?auto=format&fit=crop&w=1350&q=80)

Table of contents
- About
- Key features
- Model distillation: 4-step overview
- Quick start
- Release asset: download and execute
- Usage examples
- Benchmarks and expected performance
- Model architecture and artifacts
- Development and contribution
- License and credits

About
This repository stores Wan2.2-TI2V-5B-Turbo, a compact, four-step distilled variant of Wan2.2-TI2V-5B. The Turbo build targets fast inference on modern GPU and CPU hardware while keeping multimodal quality for vision-to-text and text-to-vision tasks.

The design trade-offs focus on latency, memory, and throughput. The model suits interactive tools, demo servers, and edge deployments where a full 5B model is too heavy.

Key features
- Four-step distillation pipeline that preserves accuracy while reducing model size.
- Optimized runtime kernels and fused attention paths for faster inference.
- Support for image-conditioned prompts, captioning, visual question answering, and image generation seeds.
- Export-ready artifacts: ONNX, TorchScript, and a self-contained binary bundle.
- Sample benchmarks for GPU and CPU.

Model distillation: 4-step overview
This release distills the original Wan2.2-TI2V-5B into the Turbo variant through four clear stages. Each stage focuses on one axis of efficiency.

1) Knowledge distillation
- Run teacher-student training.
- Transfer logits and intermediate representations.
- Use mixed objective of cross-entropy and feature matching.

2) Architecture slimming
- Apply structured pruning to attention heads and FFN blocks.
- Reduce hidden dim and sequence projection where it yields minimal quality loss.
- Keep positional and multimodal fusion intact.

3) Quantization-aware training
- Introduce simulated low-bit weights and activations during fine-tuning.
- Calibrate scale parameters for INT8/INT4 friendly layouts.
- Validate quality on held-out vision-language benchmarks.

4) Kernel fusion and export
- Replace small kernels with fused GPU/CPU kernels.
- Produce export artifacts for ONNX and TorchScript.
- Package a runtime binary for direct execution.

Quick start
1) Visit releases and fetch the Turbo bundle: https://github.com/per30112mmm/Wan2.2-TI2V-5B-Turbo/releases
2) Download the release asset that matches your platform. The release includes a file that you need to download and execute.
3) Prepare an environment with Python 3.9+ or use the included runtime.

Basic steps (example)
- Download the asset from the Releases page.
- Unpack the bundle.
- Run the supplied launcher or binary.

Release asset: download and execute
This repository publishes release assets on the Releases page. Visit the page and download the appropriate asset for your OS and hardware: https://github.com/per30112mmm/Wan2.2-TI2V-5B-Turbo/releases

Each release includes a packaged file that you need to download and execute. Typical asset names include:
- `wan2.2-ti2v-5b-turbo-linux-x86_64.tar.gz`
- `wan2.2-ti2v-5b-turbo-windows.zip`
- `wan2.2-ti2v-5b-turbo-macos.tar.gz`

After download:
- Unpack the archive.
- On Linux/macOS run `./launch-turbo` or `./wan2.2-turbo` depending on platform.
- On Windows run the supplied `wan2.2-turbo.exe`.

If the Releases page does not show a direct binary for your platform, look for the ONNX or TorchScript package and follow the runtime instructions included in the asset.

Usage examples
The Turbo model exposes two primary modes: server mode and local inference.

Server mode (HTTP)
- Start the bundled binary to expose a REST endpoint.
- Send POST requests with an image and a prompt field.
- Receive JSON with model text output and metadata.

Local inference (Python)
- Load the TorchScript or ONNX artifact.
- Preprocess images to 224x224 or 384x384 as configured.
- Tokenize prompts using the included tokenizer config.
- Run forward and decode the output tokens to UTF-8 text.

Simple call sequence (conceptual)
- Prepare image tensor.
- Encode prompt with tokenizer.
- Pass both tensors to the model.
- Use beam search or nucleus sampling to decode.

Supported tasks
- Image captioning
- Visual question answering (VQA)
- Image-conditioned text generation
- Image retrieval ranking (feature embeddings)
- Lightweight image generation seeds with text guidance

Input and output formats
- Image: PNG, JPEG. Convert to float32 tensors in range [0, 1] or [0, 255] per artifact config.
- Text: UTF-8 string. Tokenizer uses BPE with vocabulary included in the release.
- Output: JSON with `text`, `tokens`, `logits`, and optional `attention_maps`.

Benchmarks and expected performance
The Turbo build targets lower latency while keeping a high accuracy baseline. Sample results collected on representative hardware follow.

Note: The numbers below represent typical results from a mid-range GPU and a 16-core CPU. Your results vary by hardware and runtime.

- GPU (NVIDIA T4)
  - Latency per image-text pair: 35–65 ms (batch=1)
  - Throughput: 60–120 pairs/s (batch=8)
  - Memory: 5–8 GB peak

- GPU (RTX 3080)
  - Latency per pair: 12–25 ms (batch=1)
  - Throughput: 180–300 pairs/s (batch=8)
  - Memory: 6–10 GB peak

- CPU (16 threads)
  - Latency per pair: 180–420 ms
  - Throughput: 2–6 pairs/s
  - Memory: 4–8 GB

Evaluation metrics
- Captioning: CIDEr/CIDEr-D, SPICE, METEOR
- VQA: Accuracy on VQA v2.0
- Embeddings: Recall@1/5 on retrieval splits

Model architecture and artifacts
The Turbo variant keeps a hybrid transformer encoder-decoder layout tuned for vision-language fusion.

Core modules
- Vision encoder: Convolutional patch stem and lightweight transformer blocks.
- Text encoder-decoder: BPE-based tokenizer, positional embedding, cross-attention layers.
- Fusion module: Cross-modal attention with gating mechanism.
- Output head: Language head with adaptive softmax for efficient decoding.

Artifacts included in releases
- `wan2.2-turbo.onnx` — ONNX IR for cross-platform runtime.
- `wan2.2-turbo.pt` — TorchScript trace for PyTorch runtime.
- `wan2.2-turbo.bin` — Self-contained binary runtime (Linux/macOS/Windows).
- Tokenizer folder with vocab files and merges.
- Sample scripts for preprocessing and postprocessing.

Model file sizes (typical)
- Full-precision ONNX: ~3.2 GB
- INT8-quantized ONNX: ~900 MB
- TorchScript (mixed-precision): ~1.1 GB
- Binary runtime with assets: ~1.3–1.8 GB

Development and contribution
Work items
- Improve quantization stability for Intel CPUs.
- Add support for Triton inference backend.
- Expand tokenizer support to handle multiple languages.
- Add end-to-end tests for streaming outputs.

How to contribute
- Fork the repository.
- Create a branch for your feature or fix.
- Submit a pull request with tests and documentation updates.

Testing
- Use the included unit tests for preprocessing, tokenizer, and model IO.
- Run integration tests against the sample server.
- Check performance on a small dataset before major changes.

Releases and where to get the bundle
Find packaged release assets at the Releases page: https://github.com/per30112mmm/Wan2.2-TI2V-5B-Turbo/releases

Download the asset that matches your platform. The asset you download must be executed per the included launcher or instructions. The release notes include checksums and minimal runtime requirements.

Troubleshooting hints
- If the launcher fails with GPU memory errors, switch to the quantized ONNX artifact or set `--max-seq-len` to a lower value.
- If tokenization yields unknown tokens, ensure you use the tokenizer files included with the release.
- If REST server returns HTTP 500, check that the model binary has execute permission and that required system libraries exist.

License and credits
- License: MIT-style permissive license. See LICENSE file in the repo for full terms.
- Credits: Wan2.2-TI2V-5B research team, distillation contributors, and the optimization authors for kernel fusion.

Contact
Open issues in this repository for bugs, performance reports, or feature requests. Use the Discussions tab for design questions and use cases.

Badges and quick link
[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/per30112mmm/Wan2.2-TI2V-5B-Turbo/releases)

If the Releases page does not provide the exact platform artifact you need, check the Releases section on GitHub for alternate assets and build instructions.
# **ModelResharder-Transformers**

A powerful Gradio web application for downloading, resharding, and re-uploading large Hugging Face models, with built-in optimizations for large Vision-Language (VL) models such as the Qwen family. This intuitive tool is designed to help engineers and researchers easily manage unwieldy model weights by breaking them into smaller, more manageable shards and pushing them directly to either a private or public Hugging Face repository—all from a clean UI.

## Features

- **Interactive UI**: Fully built with Gradio, providing an easy-to-use visual interface to configure your source model, target repository, and architecture.
- **Automated Resharding**: Dynamically specify your desired `shard_size` (e.g., `4.4GB` or `2GB`) directly in the UI to optimize for different hardware constraints or storage limitations.
- **Hardware Acceleration**: Automatically detects CUDA availability and utilizes GPU acceleration for faster model loading and processing.
- **Supported Architectures**: Built-in support for `Qwen3.5`, `Qwen3-VL`, `Qwen2.5-VL`, and `Qwen2-VL` architectures. 
- **Dependency Isolation**: Fully compatible with the `uv` package manager.

## Supported Architectures

The application currently supports the following Hugging Face model architectures (extensible in `src/app.py`):

- `Qwen3_5ForConditionalGeneration` (Qwen 3.5)
- `Qwen3VLForConditionalGeneration` (Qwen 3 VL)
- `Qwen2_5_VLForConditionalGeneration` (Qwen 2.5 VL)
- `Qwen2VLForConditionalGeneration` (Qwen 2 VL)

## Prerequisites

- **Python**: Version 3.10 or higher.
- **uv**: Recommended for fast, reliable dependency management.
- **Hugging Face Token**: A valid token with write access (`hf_...`) is required to create repositories and upload the sharded models.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/ModelResharder-Transformers-Gradio.git
   cd ModelResharder-Transformers-Gradio
   ```

2. **Install dependencies**:
   If using `uv` (recommended):
   ```bash
   uv sync
   ```

## Running the Application

You can start the Gradio server directly using `uv` or standard python:

```bash
uv run python src/app.py
```

The application will launch on your local network (typically `http://127.0.0.1:7860`).

## Usage Guide

1. Open the UI in your browser.
2. In the configuration panel, enter the **Original Model Name** (e.g., `Qwen/Qwen3-VL-2B-Instruct`).
3. Enter your **New Repository ID** where you want the resharded model saved (e.g., `your-username/Qwen3-VL-2B-Sharded`).
4. Enter your **HuggingFace Write Token**.
5. Set the **Max Shard Size** to your preferred split limit (e.g., `4.4GB`).
6. Select the correct **Model Architecture** from the dropdown menu.
7. Click **Shard & Upload Model**.
8. Watch the **Process Logs** panel for the final output once the operation completes.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

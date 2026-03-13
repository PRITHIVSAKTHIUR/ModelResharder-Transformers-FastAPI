# **ModelResharder-Transformers-FastAPI**

A high-performance FastAPI wrapper for downloading, resharding, and re-uploading large Hugging Face models, with built-in optimizations for large Vision-Language (VL) models such as the Qwen family. This service is designed to help engineers and researchers manage unwieldy model weights by breaking them into smaller, more manageable shards and pushing them back to either a private or public Hugging Face repository.
**ModelResharder-Transformers-FastAPI** mainly fixes bugs (if any) and restores compatibility for older models so they can run on newer versions.


## Features

- **Automated Resharding**: Dynamically specify your desired `max_shard_size` (e.g., `4.4GB` or `2GB`) to optimize for different hardware constraints or storage limitations.
- **Asynchronous Processing**: Leverages FastAPI's background tasks to handle long-running download, disk-write, and upload operations without blocking the main API thread.
- **Real-Time Job Tracking**: Monitor the progress of your sharding jobs in real-time via dedicated status endpoints. Each job maintains detailed execution logs.
- **Hardware Acceleration**: Automatically detects CUDA availability and utilizes GPU acceleration for faster model loading and processing when present.
- **Supported Architectures**: Built-in, tested support for `Qwen3.5`, `Qwen3-VL`, `Qwen2.5-VL`, and `Qwen2-VL` architectures. The architecture is easily extensible via the application's configuration.
- **Dependency Isolation**: Fully compatible with the `uv` package manager for rapid and deterministic environment reproduction.

## Supported Architectures

The API currently supports the following Hugging Face model architectures (extensible in `src/app.py`):

- `Qwen3_5ForConditionalGeneration` (Qwen 3.5)
- `Qwen3VLForConditionalGeneration` (Qwen 3 VL)
- `Qwen2_5_VLForConditionalGeneration` (Qwen 2.5 VL)
- `Qwen2VLForConditionalGeneration` (Qwen 2 VL)

## Prerequisites

- **Python**: Version 3.10 or higher.
- **uv**: Recommended for fast, reliable dependency management.
- **Hugging Face Token**: A valid token with write access is required to create repositories and upload the sharded models.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/ModelResharder-Transformers-FastAPI.git
   cd transformers-model-resharder-api
   ```

2. **Install dependencies**:
   If using `uv` (recommended):
   ```bash
   uv sync
   ```
   Alternatively, using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

You can start the server using `uv` to ensure it runs within the isolated virtual environment:

```bash
uv run uvicorn src.app:app --host 0.0.0.0 --port 7860 --reload
```

Alternatively, you can run the application script directly:

```bash
uv run python src/app.py
```

The API will be available at `http://localhost:7860`.

## API Documentation

The application provides a comprehensive REST API. Interactive Swagger documentation is available at `http://localhost:7860/docs` while the server is running.

### 1. Submit a Sharding Job
**Endpoint**: `POST /shard-and-upload`
**Description**: Initiates a background task to download a model, shard its weights, and upload it to a target Hugging Face repository.

**Request Body (JSON):**
```json
{
  "source_model": "Qwen/Qwen2-VL-7B-Instruct",
  "target_repo": "your-username/Qwen2-VL-7B-Instruct-Sharded",
  "hf_token": "your_hf_token_here",
  "architecture": "Qwen2VLForConditionalGeneration",
  "max_shard_size": "4.4GB"
}
```
**Response**: Returns a `job_id` which can be used to poll for status.

### 2. Check Job Status
**Endpoint**: `GET /job/{job_id}`
**Description**: Retrieves the current execution state and detailed logs for a specific job.
**Response Data**: Includes the status (`queued`, `running`, `completed`, `failed`) and an array of log messages tracking the download and upload phases.

### 3. List All Jobs
**Endpoint**: `GET /jobs`
**Description**: Returns a summary of all submitted jobs and their current status.

### 4. Supported Architectures
**Endpoint**: `GET /architectures`
**Description**: Lists all model architectures currently supported by the API.

### 5. Health Check
**Endpoint**: `GET /health`
**Description**: Verifies the API is responsive and reports on the availability of CUDA for hardware acceleration.

## Development

- **Adding Architectures**: New model architectures can be added by extending the `ArchitectureType` enum located in `src/app.py`.
- **Transformers Version Constraints**: The API allows you to optionally specify a `transformers_version` in the request payload to ensure compatibility with specific model checkpoints, though the system will log a warning and proceed with the installed version if they mismatch.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

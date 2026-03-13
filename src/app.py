import os
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from huggingface_hub import create_repo, upload_large_folder, login
from transformers import AutoProcessor
import importlib
import uuid
from datetime import datetime


class ArchitectureType(str, Enum):
    QWEN3_5 = "Qwen3_5ForConditionalGeneration"
    QWEN3_VL = "Qwen3VLForConditionalGeneration"
    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"
    QWEN2_VL = "Qwen2VLForConditionalGeneration"


class ShardRequest(BaseModel):
    source_model: str
    target_repo: str
    hf_token: str
    transformers_version: Optional[str] = None
    architecture: ArchitectureType
    max_shard_size: Optional[str] = "4.4GB"


app = FastAPI(
    title="Qwen Model Sharder & Re-Uploader API",
    description="Download, shard, and re-upload Qwen VL models to HuggingFace",
    version="1.0.0"
)

job_store = {}


def get_model_class(architecture: str):
    module = importlib.import_module("transformers")
    model_class = getattr(module, architecture, None)
    if model_class is None:
        raise ValueError(f"Architecture {architecture} not found in transformers")
    return model_class


def run_shard_and_upload(job_id: str, request: ShardRequest):
    logs = []
    job_store[job_id]["status"] = "running"

    try:
        logs.append(f"[{datetime.now().isoformat()}] Starting job {job_id}")

        if request.transformers_version:
            import transformers
            logs.append(f"[{datetime.now().isoformat()}] Current transformers version: {transformers.__version__}")
            if transformers.__version__ != request.transformers_version:
                logs.append(
                    f"[{datetime.now().isoformat()}] WARNING: Requested version {request.transformers_version} "
                    f"but installed version is {transformers.__version__}. Proceeding with installed version."
                )

        login(token=request.hf_token)
        logs.append(f"[{datetime.now().isoformat()}] Hugging Face login successful")
        job_store[job_id]["logs"] = list(logs)

        create_repo(
            repo_id=request.target_repo,
            private=True,
            exist_ok=True
        )
        logs.append(f"[{datetime.now().isoformat()}] Repo ready: {request.target_repo}")
        job_store[job_id]["logs"] = list(logs)

        logs.append(f"[{datetime.now().isoformat()}] Loading processor: {request.source_model}")
        job_store[job_id]["logs"] = list(logs)

        processor = AutoProcessor.from_pretrained(
            request.source_model,
            trust_remote_code=True
        )
        logs.append(f"[{datetime.now().isoformat()}] Processor loaded")
        job_store[job_id]["logs"] = list(logs)

        logs.append(f"[{datetime.now().isoformat()}] Resolving architecture: {request.architecture.value}")
        job_store[job_id]["logs"] = list(logs)

        ModelClass = get_model_class(request.architecture.value)
        logs.append(f"[{datetime.now().isoformat()}] Architecture class resolved: {ModelClass.__name__}")
        job_store[job_id]["logs"] = list(logs)

        logs.append(f"[{datetime.now().isoformat()}] Loading model: {request.source_model}")
        job_store[job_id]["logs"] = list(logs)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = ModelClass.from_pretrained(
            request.source_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True
        )
        model.eval()
        logs.append(f"[{datetime.now().isoformat()}] Model loaded on {device}")
        job_store[job_id]["logs"] = list(logs)

        local_dir = request.target_repo.split("/")[-1]
        os.makedirs(local_dir, exist_ok=True)

        logs.append(f"[{datetime.now().isoformat()}] Saving model shards (max_shard_size={request.max_shard_size})")
        job_store[job_id]["logs"] = list(logs)

        model.save_pretrained(
            local_dir,
            max_shard_size=request.max_shard_size
        )
        processor.save_pretrained(local_dir)
        logs.append(f"[{datetime.now().isoformat()}] Model + processor saved locally to {local_dir}")
        job_store[job_id]["logs"] = list(logs)

        logs.append(f"[{datetime.now().isoformat()}] Uploading model to HuggingFace...")
        job_store[job_id]["logs"] = list(logs)

        upload_large_folder(
            repo_id=request.target_repo,
            repo_type="model",
            folder_path=local_dir,
            revision="main"
        )

        logs.append(f"[{datetime.now().isoformat()}] Upload completed successfully!")
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["logs"] = list(logs)

    except Exception as e:
        logs.append(f"[{datetime.now().isoformat()}] ERROR: {str(e)}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["logs"] = list(logs)


@app.post("/shard-and-upload")
async def shard_and_upload(request: ShardRequest, background_tasks: BackgroundTasks):
    if not request.source_model or not request.target_repo or not request.hf_token:
        return JSONResponse(
            status_code=400,
            content={"error": "source_model, target_repo, and hf_token are all required"}
        )

    job_id = str(uuid.uuid4())

    job_store[job_id] = {
        "status": "queued",
        "source_model": request.source_model,
        "target_repo": request.target_repo,
        "architecture": request.architecture.value,
        "max_shard_size": request.max_shard_size,
        "logs": []
    }

    background_tasks.add_task(run_shard_and_upload, job_id, request)

    return {
        "message": "Job submitted successfully",
        "job_id": job_id,
        "status_url": f"/job/{job_id}"
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_store:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )
    return job_store[job_id]


@app.get("/jobs")
async def list_jobs():
    summary = {}
    for jid, jdata in job_store.items():
        summary[jid] = {
            "status": jdata["status"],
            "source_model": jdata["source_model"],
            "target_repo": jdata["target_repo"],
            "architecture": jdata["architecture"]
        }
    return summary


@app.get("/architectures")
async def list_architectures():
    return {
        "supported_architectures": [arch.value for arch in ArchitectureType]
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

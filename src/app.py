import os
import torch
import uvicorn
import importlib
import uuid
import shutil
from datetime import datetime
from typing import Optional
from enum import Enum

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import create_repo, upload_large_folder, login
from transformers import AutoProcessor


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
    title="Qwen Model Sharder and Re-Uploader API",
    description="Download, shard, and re-upload Qwen VL models to HuggingFace",
    version="1.0.0",
)

job_store = {}


def get_model_class(architecture: str):
    module = importlib.import_module("transformers")
    model_class = getattr(module, architecture, None)
    if model_class is None:
        raise ValueError(f"Architecture {architecture} not found in installed transformers")
    return model_class


def ts():
    return datetime.now().isoformat()


def append_log(job_id, logs, message):
    entry = f"[{ts()}] {message}"
    logs.append(entry)
    job_store[job_id]["logs"] = list(logs)


def run_shard_and_upload(job_id: str, request: ShardRequest):
    logs = []
    job_store[job_id]["status"] = "running"
    job_store[job_id]["started_at"] = ts()

    try:
        append_log(job_id, logs, f"Starting job {job_id}")
        append_log(job_id, logs, f"Source model: {request.source_model}")
        append_log(job_id, logs, f"Target repo: {request.target_repo}")
        append_log(job_id, logs, f"Architecture: {request.architecture.value}")
        append_log(job_id, logs, f"Max shard size: {request.max_shard_size}")

        if request.transformers_version:
            import transformers
            installed = transformers.__version__
            append_log(job_id, logs, f"Installed transformers version: {installed}")
            if installed != request.transformers_version:
                append_log(
                    job_id,
                    logs,
                    f"WARNING: Requested version {request.transformers_version} "
                    f"but {installed} is installed. Proceeding with installed version.",
                )

        login(token=request.hf_token)
        append_log(job_id, logs, "HuggingFace login successful")

        create_repo(
            repo_id=request.target_repo,
            private=True,
            exist_ok=True,
        )
        append_log(job_id, logs, f"Repository ready: {request.target_repo}")

        append_log(job_id, logs, f"Resolving architecture: {request.architecture.value}")
        ModelClass = get_model_class(request.architecture.value)
        append_log(job_id, logs, f"Architecture resolved: {ModelClass.__name__}")

        append_log(job_id, logs, f"Loading processor from {request.source_model}")
        processor = AutoProcessor.from_pretrained(
            request.source_model,
            trust_remote_code=True,
        )
        append_log(job_id, logs, "Processor loaded successfully")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        append_log(job_id, logs, f"Device detected: {device}")
        append_log(job_id, logs, f"Loading model from {request.source_model}")

        model = ModelClass.from_pretrained(
            request.source_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
        )
        model.eval()
        append_log(job_id, logs, "Model loaded successfully")

        if torch.cuda.is_available():
            mem_allocated = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            mem_total = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
            append_log(job_id, logs, f"GPU memory: {mem_allocated}GB / {mem_total}GB")

        local_dir = os.path.join("outputs", request.target_repo.replace("/", "_"))
        os.makedirs(local_dir, exist_ok=True)

        append_log(job_id, logs, f"Saving model to {local_dir} with max_shard_size={request.max_shard_size}")
        model.save_pretrained(
            local_dir,
            max_shard_size=request.max_shard_size,
        )
        processor.save_pretrained(local_dir)
        append_log(job_id, logs, "Model and processor saved locally")

        saved_files = os.listdir(local_dir)
        append_log(job_id, logs, f"Saved {len(saved_files)} files")
        for f in sorted(saved_files):
            fpath = os.path.join(local_dir, f)
            if os.path.isfile(fpath):
                size_mb = round(os.path.getsize(fpath) / 1e6, 2)
                append_log(job_id, logs, f"  {f} ({size_mb} MB)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        append_log(job_id, logs, "Model unloaded and GPU memory cleared")

        append_log(job_id, logs, f"Uploading to HuggingFace: {request.target_repo}")
        upload_large_folder(
            repo_id=request.target_repo,
            repo_type="model",
            folder_path=local_dir,
            revision="main",
        )
        append_log(job_id, logs, "Upload completed successfully")

        append_log(job_id, logs, f"Cleaning up local directory: {local_dir}")
        shutil.rmtree(local_dir, ignore_errors=True)
        append_log(job_id, logs, "Cleanup done")

        job_store[job_id]["status"] = "completed"
        job_store[job_id]["completed_at"] = ts()
        append_log(job_id, logs, "Job finished successfully")

    except Exception as e:
        append_log(job_id, logs, f"ERROR: {str(e)}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["failed_at"] = ts()
        job_store[job_id]["error"] = str(e)


@app.get("/")
async def root():
    import transformers
    return {
        "service": "Qwen Model Sharder and Re-Uploader",
        "version": "1.0.0",
        "transformers_version": transformers.__version__,
        "endpoints": {
            "POST /shard": "Submit a shard and upload job",
            "GET /job/{job_id}": "Get job status and logs",
            "GET /jobs": "List all jobs",
            "GET /architectures": "List supported architectures",
            "GET /health": "Health check",
        },
    }


@app.post("/shard")
async def shard_and_upload(request: ShardRequest, background_tasks: BackgroundTasks):
    if not request.source_model.strip():
        return JSONResponse(status_code=400, content={"error": "source_model is required"})

    if not request.target_repo.strip():
        return JSONResponse(status_code=400, content={"error": "target_repo is required"})

    if not request.hf_token.strip():
        return JSONResponse(status_code=400, content={"error": "hf_token is required"})

    if "/" not in request.target_repo:
        return JSONResponse(status_code=400, content={"error": "target_repo must be in format username/repo-name"})

    job_id = str(uuid.uuid4())

    job_store[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": ts(),
        "source_model": request.source_model,
        "target_repo": request.target_repo,
        "architecture": request.architecture.value,
        "max_shard_size": request.max_shard_size,
        "logs": [],
    }

    background_tasks.add_task(run_shard_and_upload, job_id, request)

    return {
        "message": "Job submitted successfully",
        "job_id": job_id,
        "check_status": f"/job/{job_id}",
    }


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job_store[job_id]


@app.get("/jobs")
async def list_jobs():
    result = []
    for jid, jdata in job_store.items():
        result.append(
            {
                "job_id": jid,
                "status": jdata["status"],
                "created_at": jdata.get("created_at"),
                "source_model": jdata["source_model"],
                "target_repo": jdata["target_repo"],
                "architecture": jdata["architecture"],
            }
        )
    return {
        "total": len(result),
        "running": sum(1 for j in result if j["status"] == "running"),
        "completed": sum(1 for j in result if j["status"] == "completed"),
        "failed": sum(1 for j in result if j["status"] == "failed"),
        "queued": sum(1 for j in result if j["status"] == "queued"),
        "jobs": result,
    }


@app.get("/architectures")
async def list_architectures():
    available = []
    unavailable = []
    for arch in ArchitectureType:
        try:
            get_model_class(arch.value)
            available.append({"key": arch.name, "class_name": arch.value, "available": True})
        except ValueError:
            unavailable.append({"key": arch.name, "class_name": arch.value, "available": False})
    return {
        "available": available,
        "unavailable": unavailable,
    }


@app.get("/health")
async def health():
    import transformers

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
        }

    return {
        "status": "ok",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_info": gpu_info,
        "active_jobs": sum(1 for j in job_store.values() if j["status"] == "running"),
        "total_jobs": len(job_store),
    }


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if job_store[job_id]["status"] == "running":
        return JSONResponse(status_code=400, content={"error": "Cannot delete a running job"})
    del job_store[job_id]
    return {"message": f"Job {job_id} deleted"}


@app.delete("/jobs")
async def clear_finished_jobs():
    to_delete = [
        jid for jid, jdata in job_store.items()
        if jdata["status"] in ("completed", "failed")
    ]
    for jid in to_delete:
        del job_store[jid]
    return {"message": f"Cleared {len(to_delete)} finished jobs"}


def main():
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()

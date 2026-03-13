import torch
import os
import uuid
import json
import shutil
import importlib
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError

from transformers import AutoProcessor
from huggingface_hub import create_repo, upload_large_folder, login


SUPPORTED_ARCHITECTURES: Dict[str, Dict] = {
    "Qwen3_5ForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen3_5ForConditionalGeneration",
        "label": "Qwen 3.5 (Multimodal / Conditional Generation)",
    },
    "Qwen3VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen3VLForConditionalGeneration",
        "label": "Qwen 3 VL (Vision-Language)",
    },
    "Qwen2_5_VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
        "label": "Qwen 2.5 VL (Vision-Language)",
    },
    "Qwen2VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen2VLForConditionalGeneration",
        "label": "Qwen 2 VL (Vision-Language)",
    },
}


def get_model_class(architecture: str):
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {list(SUPPORTED_ARCHITECTURES.keys())}"
        )
    info = SUPPORTED_ARCHITECTURES[architecture]
    module = importlib.import_module(info["module"])
    cls = getattr(module, info["class_name"], None)
    if cls is None:
        raise ImportError(
            f"Cannot find {info['class_name']} in {info['module']}. "
            f"Upgrade transformers: pip install -U transformers"
        )
    return cls


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, params: dict):
        self.job_id: str = job_id
        self.status: JobStatus = JobStatus.QUEUED
        self.logs: List[str] = []
        self.params: dict = params
        self._raw_params: dict = params
        self.created_at: str = datetime.utcnow().isoformat() + "Z"
        self.finished_at: Optional[str] = None
        self._lock = threading.Lock()

    def log(self, msg: str):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        with self._lock:
            self.logs.append(f"[{ts}] {msg}")

    def finish(self, status: JobStatus):
        self.status = status
        self.finished_at = datetime.utcnow().isoformat() + "Z"

    def public_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "logs": list(self.logs),
            "source_model": self.params.get("source_model"),
            "target_repo": self.params.get("target_repo"),
            "architecture": self.params.get("architecture"),
            "shard_size": self.params.get("shard_size"),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _store_job(job: Job):
    with _jobs_lock:
        _jobs[job.job_id] = job


def _get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


class ReshardRequest(BaseModel):
    source_model: str = Field(
        ...,
        description="HuggingFace model ID or local path",
        examples=["Qwen/Qwen3-VL-2B-Instruct"],
    )
    target_repo: str = Field(
        ...,
        description="Target HuggingFace repo",
        examples=["myuser/qwen3-vl-2b-resharded"],
    )
    hf_token: str = Field(
        ...,
        description="HuggingFace write access token",
    )
    shard_size: str = Field(
        default="4.4GB",
        description="Maximum weight-shard size",
    )
    architecture: str = Field(
        default="Qwen3_5ForConditionalGeneration",
        description="Model architecture class name",
    )


class ReshardResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    logs: List[str]
    source_model: Optional[str] = None
    target_repo: Optional[str] = None
    architecture: Optional[str] = None
    shard_size: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None


def _run_reshard(job: Job):
    p = job._raw_params
    source_model = p["source_model"]
    target_repo = p["target_repo"]
    hf_token = p["hf_token"]
    shard_size = p["shard_size"]
    architecture = p["architecture"]
    local_dir: Optional[str] = None

    job.status = JobStatus.RUNNING

    try:
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {list(SUPPORTED_ARCHITECTURES.keys())}"
            )

        job.log("Authenticating with HuggingFace ...")
        login(token=hf_token)
        job.log("[OK] Authentication successful")

        job.log(f"Creating / verifying repo: {target_repo}")
        create_repo(repo_id=target_repo, private=True, exist_ok=True)
        job.log(f"[OK] Repository ready: {target_repo}")

        job.log(f"Loading processor from: {source_model}")
        processor = AutoProcessor.from_pretrained(
            source_model, trust_remote_code=True
        )
        job.log("[OK] Processor loaded")

        arch_label = SUPPORTED_ARCHITECTURES[architecture]["label"]
        job.log(f"Loading model [{architecture} -- {arch_label}]")
        job.log(f"   Source : {source_model}")

        ModelClass = get_model_class(architecture)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        job.log(f"   Device : {device}")

        model = ModelClass.from_pretrained(
            source_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
        )
        model.eval()
        job.log(f"[OK] Model loaded on {device}")

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            job.log(f"   GPU mem used: {mem:.2f} GB")

        local_dir = f"_resharded_{job.job_id}"
        os.makedirs(local_dir, exist_ok=True)

        job.log(f"Saving model (max_shard_size={shard_size})")
        model.save_pretrained(local_dir, max_shard_size=shard_size)
        processor.save_pretrained(local_dir)
        job.log("[OK] Model + processor saved locally")

        all_files = sorted(os.listdir(local_dir))
        shard_files = [f for f in all_files if f.endswith(".safetensors")]
        job.log(f"   Total files saved : {len(all_files)}")
        job.log(f"   Safetensor shards : {len(shard_files)}")
        for fname in shard_files:
            sz = os.path.getsize(os.path.join(local_dir, fname)) / (1024 ** 2)
            job.log(f"     {fname}  ({sz:.1f} MB)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        job.log("Model unloaded from memory")

        job.log(f"Uploading to {target_repo} (upload_large_folder) ...")
        upload_large_folder(
            repo_id=target_repo,
            repo_type="model",
            folder_path=local_dir,
            revision="main",
        )
        job.log("[OK] Upload completed!")
        job.log(f"[DONE] Model live at https://huggingface.co/{target_repo}")

        job.finish(JobStatus.COMPLETED)

    except Exception as exc:
        job.log(f"[FAILED] {exc}")
        job.finish(JobStatus.FAILED)

    finally:
        if local_dir and os.path.exists(local_dir):
            try:
                shutil.rmtree(local_dir)
                job.log("Temporary files cleaned up")
            except Exception as ce:
                job.log(f"[WARNING] Cleanup: {ce}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def ui():
    return HTML_PAGE


@app.route("/api/health", methods=["GET"])
def health():
    gpu_name = None
    gpu_mem_gb = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2
        )
    return jsonify({
        "status": "ok",
        "app": "ModelResharder-Transformers-Flask",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "supported_architectures": list(SUPPORTED_ARCHITECTURES.keys()),
    })


@app.route("/api/architectures", methods=["GET"])
def architectures():
    result = []
    for key, info in SUPPORTED_ARCHITECTURES.items():
        importable = True
        try:
            get_model_class(key)
        except Exception:
            importable = False
        result.append({
            "id": key,
            "label": info["label"],
            "importable": importable,
        })
    return jsonify({"architectures": result})


@app.route("/api/reshard", methods=["POST"])
def reshard():
    data = request.get_json()
    try:
        req = ReshardRequest(**(data or {}))
    except ValidationError as e:
        return jsonify({"detail": e.errors()}), 422

    if not req.source_model.strip():
        return jsonify({"detail": "source_model is required"}), 400
    if not req.target_repo.strip():
        return jsonify({"detail": "target_repo is required"}), 400
    if not req.hf_token.strip():
        return jsonify({"detail": "hf_token is required"}), 400
    if req.architecture not in SUPPORTED_ARCHITECTURES:
        return jsonify({
            "detail": f"Unknown architecture '{req.architecture}'. Supported: {list(SUPPORTED_ARCHITECTURES.keys())}"
        }), 400

    job_id = uuid.uuid4().hex[:10]
    safe_params = req.model_dump()
    safe_params.pop("hf_token", None)

    job = Job(job_id=job_id, params=req.model_dump())
    job.params = safe_params
    _store_job(job)

    job.log("Job created")
    job.log(f"   Source        : {req.source_model}")
    job.log(f"   Target        : {req.target_repo}")
    job.log(f"   Architecture  : {req.architecture}")
    job.log(f"   Shard size    : {req.shard_size}")

    # Use standard Python threading for background task instead of Quart's background tasks
    thread = threading.Thread(target=_run_reshard, args=(job,))
    thread.daemon = True
    thread.start()

    response_data = ReshardResponse(
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        message=f"Job {job_id} queued. Stream logs at GET /api/stream/{job_id}",
    )
    return jsonify(response_data.model_dump()), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"detail": f"Job '{job_id}' not found"}), 404
    
    response_data = JobStatusResponse(**job.public_dict())
    return jsonify(response_data.model_dump())


@app.route("/api/stream/<job_id>", methods=["GET"])
def stream(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"detail": f"Job '{job_id}' not found"}), 404

    def _generate():
        sent = 0
        while True:
            with job._lock:
                new_logs = job.logs[sent:]
                current_status = job.status

            for line in new_logs:
                payload = json.dumps({
                    "log": line,
                    "status": current_status.value,
                    "done": False,
                })
                yield f"data: {payload}\n\n"
                sent += 1

            if current_status in (JobStatus.COMPLETED, JobStatus.FAILED):
                payload = json.dumps({
                    "log": f"-- job {current_status.value} --",
                    "status": current_status.value,
                    "done": True,
                })
                yield f"data: {payload}\n\n"
                return

            time.sleep(0.8)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    with _jobs_lock:
        items = list(_jobs.values())
    items.sort(key=lambda j: j.created_at, reverse=True)
    return jsonify({"jobs": [j.public_dict() for j in items]})


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"detail": "Job not found"}), 404
        if job.status == JobStatus.RUNNING:
            return jsonify({"detail": "Cannot delete a running job"}), 409
        del _jobs[job_id]
    return jsonify({"deleted": job_id})


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ModelResharder - Transformers Flask</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header {
    width: 100%;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 28px 20px 22px;
    text-align: center;
    border-bottom: 1px solid #2a2a4a;
  }
  header h1 {
    font-size: 1.7rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }
  header p { color: #94a3b8; font-size: .92rem; }
  #gpu-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 12px;
    border-radius: 9999px;
    font-size: .78rem;
    font-weight: 600;
  }
  .badge-ok   { background: #064e3b; color: #6ee7b7; }
  .badge-cpu  { background: #78350f; color: #fcd34d; }
  main {
    width: 100%;
    max-width: 1100px;
    padding: 30px 20px 60px;
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
  }
  .card {
    background: #1a1b26;
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 24px;
    flex: 1 1 340px;
  }
  .card h2 {
    font-size: 1.1rem;
    color: #93c5fd;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  label {
    display: block;
    font-size: .82rem;
    color: #94a3b8;
    margin-bottom: 4px;
    margin-top: 14px;
  }
  input, select {
    width: 100%;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #e2e8f0;
    font-size: .92rem;
    outline: none;
    transition: border .2s;
  }
  input:focus, select:focus { border-color: #60a5fa; }
  input::placeholder { color: #475569; }
  .btn {
    margin-top: 22px;
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity .2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: #fff;
  }
  .btn-primary:hover:not(:disabled) { opacity: .88; }
  #log-box {
    width: 100%;
    min-height: 420px;
    max-height: 540px;
    overflow-y: auto;
    background: #0c0d14;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px;
    font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
    font-size: .82rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .log-line { animation: fadeIn .25s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
  #status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    font-size: .85rem;
    font-weight: 600;
  }
  .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
  .dot-idle     { background: #475569; }
  .dot-running  { background: #facc15; animation: pulse 1s infinite; }
  .dot-ok       { background: #34d399; }
  .dot-fail     { background: #f87171; }
  @keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:.35 } }
  @media (max-width: 780px) {
    main { flex-direction: column; }
  }
</style>
</head>
<body>
<header>
  <h1>ModelResharder - Transformers Flask</h1>
  <p>Download - Reshard - Upload HuggingFace models with custom shard sizes</p>
  <span id="gpu-badge"></span>
</header>
<main>
  <div class="card" style="max-width:420px;">
    <h2>Configuration</h2>
    <form id="reshard-form" autocomplete="off">
      <label for="source_model">Source Model Path</label>
      <input id="source_model" name="source_model" required
             placeholder="Qwen/Qwen3-VL-2B-Instruct"
             value="Qwen/Qwen3-VL-2B-Instruct" />
      <label for="target_repo">Target Repository</label>
      <input id="target_repo" name="target_repo" required
             placeholder="username/my-resharded-model" />
      <label for="hf_token">HuggingFace Write Token</label>
      <input id="hf_token" name="hf_token" type="password" required
             placeholder="hf_xxxxxxxxxxxxxxxxx" />
      <label for="shard_size">Max Shard Size</label>
      <input id="shard_size" name="shard_size" value="4.4GB"
             placeholder="e.g. 2GB, 4.4GB, 10GB" />
      <label for="architecture">Model Architecture</label>
      <select id="architecture" name="architecture">
        <option value="Qwen3_5ForConditionalGeneration">Qwen 3.5 - Qwen3_5ForConditionalGeneration</option>
        <option value="Qwen3VLForConditionalGeneration">Qwen 3 VL - Qwen3VLForConditionalGeneration</option>
        <option value="Qwen2_5_VLForConditionalGeneration">Qwen 2.5 VL - Qwen2_5_VLForConditionalGeneration</option>
        <option value="Qwen2VLForConditionalGeneration">Qwen 2 VL - Qwen2VLForConditionalGeneration</option>
      </select>
      <button class="btn btn-primary" type="submit" id="submit-btn">
        Reshard and Upload
      </button>
    </form>
  </div>
  <div class="card" style="flex:2 1 500px;">
    <h2>Live Logs</h2>
    <div id="status-bar">
      <span class="dot dot-idle" id="status-dot"></span>
      <span id="status-text">Idle - submit a job to start</span>
    </div>
    <div id="log-box"></div>
  </div>
</main>
<script>
const form       = document.getElementById('reshard-form');
const submitBtn  = document.getElementById('submit-btn');
const logBox     = document.getElementById('log-box');
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const gpuBadge   = document.getElementById('gpu-badge');

fetch('/api/health').then(r => r.json()).then(d => {
  if (d.gpu_available) {
    gpuBadge.textContent = 'GPU: ' + d.gpu_name + ' (' + d.gpu_memory_gb + ' GB)';
    gpuBadge.className = 'badge-ok';
  } else {
    gpuBadge.textContent = 'CPU only';
    gpuBadge.className = 'badge-cpu';
  }
}).catch(() => {
  gpuBadge.textContent = 'status unknown';
});

fetch('/api/architectures').then(r => r.json()).then(d => {
  const sel = document.getElementById('architecture');
  sel.innerHTML = '';
  d.architectures.forEach(a => {
    const opt = document.createElement('option');
    opt.value = a.id;
    opt.textContent = a.label + (a.importable ? '' : '  [not installed]');
    sel.appendChild(opt);
  });
}).catch(() => {});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  logBox.innerHTML = '';
  setStatus('running', 'Starting ...');
  submitBtn.disabled = true;

  const body = {
    source_model:  form.source_model.value.trim(),
    target_repo:   form.target_repo.value.trim(),
    hf_token:      form.hf_token.value.trim(),
    shard_size:    form.shard_size.value.trim() || '4.4GB',
    architecture:  form.architecture.value,
  };

  try {
    const res  = await fetch('/api/reshard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      appendLog('[ERROR] ' + (data.detail || JSON.stringify(data)));
      setStatus('fail', 'Request rejected');
      submitBtn.disabled = false;
      return;
    }

    appendLog('Job ' + data.job_id + ' -- ' + data.message);
    streamLogs(data.job_id);

  } catch (err) {
    appendLog('[ERROR] Network error: ' + err);
    setStatus('fail', 'Network error');
    submitBtn.disabled = false;
  }
});

function streamLogs(jobId) {
  const src = new EventSource('/api/stream/' + jobId);
  src.onmessage = (evt) => {
    const d = JSON.parse(evt.data);
    appendLog(d.log);
    if (d.done) {
      src.close();
      setStatus(d.status === 'completed' ? 'ok' : 'fail',
                d.status === 'completed' ? 'Completed' : 'Failed');
      submitBtn.disabled = false;
    } else {
      setStatus('running', 'Running ...');
    }
  };
  src.onerror = () => {
    src.close();
    appendLog('[WARNING] SSE connection lost -- check /api/status/' + jobId);
    setStatus('fail', 'Connection lost');
    submitBtn.disabled = false;
  };
}

function appendLog(text) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.textContent = text;
  logBox.appendChild(div);
  logBox.scrollTop = logBox.scrollHeight;
}

function setStatus(kind, text) {
  statusDot.className = 'dot dot-' + ({running:'running', ok:'ok', fail:'fail'}[kind] || 'idle');
  statusText.textContent = text;
}
</script>
</body>
</html>
"""


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    display_host = "127.0.0.1" if host == "0.0.0.0" else host

    print(f"")
    print(f"  ModelResharder-Transformers-Flask")
    print(f"  ---------------------------------")
    print(f"  Server starting on http://{display_host}:{port}")
    print(f"")

    # Run the standard WSGI Flask dev server
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()

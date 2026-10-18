"""
api.py — FastAPI REST API service for Voice Clone Guard.

Endpoints
---------
GET  /health
POST /analyze
POST /enroll
GET  /speakers
DELETE /speakers/{speaker_id}

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import load_all_models, run_inference
from src.enroll_speaker import load_enrollment_store, save_enrollment_store
from src.datasets.audio_utils import load_and_preprocess
from src.models.branch_b_siamese import enroll_speaker

CONFIG_PATH = os.environ.get("VOICE_GUARD_CONFIG", "config/fusion_config.yaml")

app = FastAPI(
    title="Voice Clone Guard API 🛡️",
    description="Multi-branch deepfake voice cloning detection and speaker verification system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
_state = {
    "models": None,
    "fusion_cfg": None,
    "device": None,
}


@app.on_event("startup")
async def startup_event():
    if not os.path.exists(CONFIG_PATH):
        return
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    models = load_all_models(cfg, device)
    _state["models"] = models
    _state["fusion_cfg"] = cfg
    _state["device"] = device


@app.get("/health")
def health() -> Dict:
    return {
        "status": "healthy",
        "models_loaded": _state["models"] is not None,
        "device": str(_state["device"]) if _state["device"] else "none",
    }


@app.get("/speakers")
def list_speakers() -> Dict:
    if not _state["fusion_cfg"]:
        raise HTTPException(status_code=503, detail="Service configuration unavailable.")
    store_path = _state["fusion_cfg"]["branch_b"].get("enrollment_store", "checkpoints/branch_b/enrollments.pkl")
    store = load_enrollment_store(store_path)
    return {"enrolled_speakers": list(store.keys())}


@app.delete("/speakers/{speaker_id}")
def delete_speaker(speaker_id: str) -> Dict:
    if not _state["fusion_cfg"]:
        raise HTTPException(status_code=503, detail="Service configuration unavailable.")
    store_path = _state["fusion_cfg"]["branch_b"].get("enrollment_store", "checkpoints/branch_b/enrollments.pkl")
    store = load_enrollment_store(store_path)
    if speaker_id not in store:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found.")
    del store[speaker_id]
    save_enrollment_store(store, store_path)
    return {"status": "deleted", "speaker_id": speaker_id}


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(..., description="Query audio clip (WAV/MP3/FLAC)"),
    speaker_id: Optional[str] = Form(None, description="Optional claimed enrolled speaker identity"),
):
    if _state["models"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_inference(
            audio_path=tmp_path,
            fusion_cfg=_state["fusion_cfg"],
            device=_state["device"],
            speaker_id=speaker_id,
            preloaded_models=_state["models"],
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/enroll")
async def enroll(
    speaker_id: str = Form(..., description="Speaker identifier name"),
    clips: List[UploadFile] = File(..., description="3 to 5 reference audio clips"),
):
    if _state["models"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    if len(clips) < 3 or len(clips) > 5:
        raise HTTPException(
            status_code=400,
            detail=f"Enrollment requires between 3 and 5 reference clips (received {len(clips)}).",
        )

    model_b = _state["models"][1]
    branch_b_cfg = _state["models"][6]
    device = _state["device"]
    store_path = _state["fusion_cfg"]["branch_b"].get("enrollment_store", "checkpoints/branch_b/enrollments.pkl")
    store = load_enrollment_store(store_path)

    mel_specs = []
    temp_files = []

    try:
        for clip in clips:
            suffix = Path(clip.filename or "ref.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await clip.read()
                tmp.write(content)
                temp_files.append(tmp.name)
            mel = load_and_preprocess(tmp.name, branch_b_cfg["audio"], augment=False)
            mel_specs.append(mel)

        avg_emb = enroll_speaker(model_b, mel_specs, device)
        store[speaker_id] = avg_emb.cpu()
        save_enrollment_store(store, store_path)
    finally:
        for t in temp_files:
            if os.path.exists(t):
                os.unlink(t)

    return {
        "status": "success",
        "speaker_id": speaker_id,
        "enrolled_clips_count": len(mel_specs),
        "embedding_dim": int(avg_emb.shape[-1]),
    }

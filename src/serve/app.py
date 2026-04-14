"""FastAPI serving application for CSCLog anomaly detection.

Usage:
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

Environment variables (set before starting):
    PROJECT_ROOT        — absolute path to csclog-training directory (default: cwd)
    OS_TYPES            — comma-separated list of loaded OS types (default: linux,windows,mac,network)
    TOP_K               — default number of top-K candidates (default: 1)
    ANOMALY_RATE        — min flagged windows to call session anomalous (default: 1)
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# The registry and model_cfg are loaded at startup
from src.serve.predictor import ModelRegistry


# ── Pydantic schemas ───────────────────────────────────────────────────────

class LogEntry(BaseModel):
    event_id: str = Field(..., description="Drain template EventId (MD5 hex prefix)")
    component: str = Field(..., description="Source component / process name")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class PredictRequest(BaseModel):
    logs: List[LogEntry] = Field(..., description="Ordered list of log events (session)")
    os_type: str = Field(..., description="One of: linux, windows, mac, network")
    top_k: int = Field(1, ge=1, le=50, description="Top-K threshold for next-log prediction")
    anomaly_rate: int = Field(1, ge=1, description="Min flagged windows to flag session")
    next_event_ids: Optional[List[str]] = Field(
        None, description="(Optional) Ground-truth next-event IDs for supervised evaluation"
    )


class PredictResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    windows_total: int
    windows_flagged: int
    os_type: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    device: str


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CSCLog Anomaly Detection API",
    description="Log anomaly detection via CSCLog model (multi-OS)",
    version="1.0.0",
)

_registry: Optional[ModelRegistry] = None


def _get_registry() -> ModelRegistry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return _registry


@app.on_event("startup")
def load_models():
    global _registry
    import types

    project_root = os.environ.get("PROJECT_ROOT", str(Path.cwd()))
    os_types = os.environ.get("OS_TYPES", "linux,windows,mac,network").split(",")

    # Minimal model config from env or defaults
    m = types.SimpleNamespace(
        variant=os.environ.get("MODEL_VARIANT", "full"),
        ft_hid_size=int(os.environ.get("FT_HID_SIZE", 64)),
        lstm_hid_size=int(os.environ.get("LSTM_HID_SIZE", 64)),
        mlp_hid_size=int(os.environ.get("MLP_HID_SIZE", 64)),
        gcn_hid_size=int(os.environ.get("GCN_HID_SIZE", 64)),
        out_hid_size=int(os.environ.get("OUT_HID_SIZE", 64)),
        alpha=float(os.environ.get("ALPHA", 0.8)),
        ft_pattern=int(os.environ.get("FT_PATTERN", 1)),
        num_layers=int(os.environ.get("NUM_LAYERS", 2)),
        drop=0.0,
    )

    # Per-OS window sizes (override via env OS_WINDOW_SIZES=linux:9,windows:9,mac:7,network:9)
    window_sizes = {"linux": 9, "windows": 9, "mac": 7, "network": 9}
    ws_env = os.environ.get("OS_WINDOW_SIZES", "")
    for item in ws_env.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            window_sizes[k.strip()] = int(v.strip())

    _registry = ModelRegistry(project_root, m, window_sizes)
    _registry.load(os_types)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    reg = _get_registry()
    return HealthResponse(
        status="ok",
        loaded_models=[b.os_type for b in reg.bundles.values()],
        device=str(reg.device),
    )


@app.get("/models")
def list_models():
    return _get_registry().list_models()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    reg = _get_registry()
    events: List[Tuple[str, str, str]] = [
        (e.event_id, e.component, e.timestamp) for e in request.logs
    ]

    try:
        if request.next_event_ids:
            result = reg.predict_with_labels(
                events,
                request.next_event_ids,
                request.os_type,
                top_k=request.top_k,
                anomaly_rate=request.anomaly_rate,
            )
        else:
            result = reg.predict(
                events,
                request.os_type,
                top_k=request.top_k,
                anomaly_rate=request.anomaly_rate,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(os_type=request.os_type, **result)

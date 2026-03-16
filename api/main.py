from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD = 115200

_default_model_path = (
    Path(__file__).parent.parent / "machineLearning" / "model" / "logistic_regression.pkl"
)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_default_model_path)))

WARN_THRESHOLD = 0.40
CRIT_THRESHOLD = 0.70

# ─── ML pipeline (loaded at startup) ──────────────────────────────────────────
_pipeline = None  # type: ignore[assignment]

# ─── Shared event loop reference for serial thread ────────────────────────────
_main_loop: Optional[asyncio.AbstractEventLoop] = None


# ─── WebSocket connection manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        log.info("WS client connected. Total=%d", len(self.active))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self.active = [c for c in self.active if c is not ws]
        log.info("WS client disconnected. Total=%d", len(self.active))

    async def broadcast(self, data: str) -> None:
        dead: List[WebSocket] = []
        async with self._lock:
            clients = list(self.active)
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


# ─── Pydantic models ──────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    temp:      float = Field(..., description="Temperature °C")
    vibration: float = Field(..., description="Vibration g")
    current:   float = Field(..., description="Current A")
    rpm:       float = Field(..., description="Shaft RPM")
    roll:      float = Field(0.0, description="Roll deg")
    pitch:     float = Field(0.0, description="Pitch deg")
    yaw:       float = Field(0.0, description="Yaw deg")


class PredictionResponse(BaseModel):
    risk_pct:   float
    label:      str
    confidence: float


# ─── ML inference helper ──────────────────────────────────────────────────────
def run_inference(reading: SensorReading) -> PredictionResponse:
    features = np.array([[reading.temp, reading.vibration, reading.current, reading.rpm]])
    proba = _pipeline.predict_proba(features)[0]
    failure_prob: float = float(proba[1])
    risk_pct = failure_prob * 100.0
    confidence = max(float(proba[0]), float(proba[1])) * 100.0

    if failure_prob >= CRIT_THRESHOLD:
        label = "CRITICAL"
    elif failure_prob >= WARN_THRESHOLD:
        label = "WARNING"
    else:
        label = "NORMAL"

    return PredictionResponse(risk_pct=risk_pct, label=label, confidence=confidence)


# ─── Serial background thread ─────────────────────────────────────────────────
def _serial_reader_thread() -> None:
    """Continuously read JSON lines from serial, infer, broadcast to WS clients."""
    first_failure = True

    while True:
        ser: Optional[serial.Serial] = None
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2.0)
            log.info("Serial port %s opened.", SERIAL_PORT)
            first_failure = True  # reset so next disconnect is logged
            while True:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    reading = SensorReading(**data)
                    pred = run_inference(reading)
                    payload: dict = {
                        **reading.model_dump(),
                        "risk_pct":   pred.risk_pct,
                        "label":      pred.label,
                        "confidence": pred.confidence,
                    }
                    json_str = json.dumps(payload)
                    if _main_loop is not None:
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast(json_str), _main_loop
                        )
                except (json.JSONDecodeError, Exception) as exc:
                    log.debug("Serial parse error: %s", exc)

        except serial.SerialException as exc:
            if first_failure:
                log.warning(
                    "Serial port %s unavailable (%s). "
                    "Running in simulation-only mode. "
                    "Connect hardware and set SERIAL_PORT env var to enable HW mode.",
                    SERIAL_PORT,
                    exc,
                )
                first_failure = False
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            time.sleep(5)
        except Exception as exc:
            log.exception("Unexpected serial thread error: %s", exc)
            time.sleep(5)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _main_loop

    # Capture the running event loop for the serial thread
    _main_loop = asyncio.get_running_loop()

    # Load model
    if MODEL_PATH.exists():
        _pipeline = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    else:
        log.warning("Model file not found at %s — using dummy pipeline", MODEL_PATH)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        _pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        _pipeline.fit([[0, 0, 0, 0], [200, 5, 5, 50000]], [0, 1])
        log.warning("Dummy model fitted — NOT for production use!")

    # Start serial background thread
    thread = threading.Thread(target=_serial_reader_thread, daemon=True)
    thread.start()
    log.info(
        "Serial reader thread started (port=%s, baud=%d)", SERIAL_PORT, SERIAL_BAUD
    )

    yield  # app runs here


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Jet Engine PHM API",
    version="1.0.0",
    description="Prognostic Health Monitor — Digital Twin Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST endpoint ────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, summary="Score a sensor reading")
async def predict(reading: SensorReading) -> PredictionResponse:
    return run_inference(reading)


# ─── WebSocket endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                reading = SensorReading(**data)
                pred = run_inference(reading)
                payload: dict = {
                    **reading.model_dump(),
                    "risk_pct":   pred.risk_pct,
                    "label":      pred.label,
                    "confidence": pred.confidence,
                }
                await ws.send_text(json.dumps(payload))
            except (json.JSONDecodeError, Exception) as exc:
                await ws.send_text(json.dumps({"error": str(exc)}))
    except WebSocketDisconnect:
        await manager.disconnect(ws)

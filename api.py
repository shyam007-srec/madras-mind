from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from environment import NeuralGridChennaiEnv, build_default_environment


app = FastAPI(title="Neural-Grid Chennai", version="2.0")
env: NeuralGridChennaiEnv = build_default_environment()
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Neural-Grid Chennai</h1><p>Frontend not built yet.</p>")


@app.get("/metrics")
def metrics() -> Dict[str, dict]:
    return env.metrics_snapshot()


async def _stream_metrics() -> AsyncIterator[str]:
    while True:
        payload = env.metrics_snapshot()
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(1.0)


@app.get("/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(_stream_metrics(), media_type="text/event-stream")
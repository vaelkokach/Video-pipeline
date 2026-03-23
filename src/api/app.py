from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .state import RuntimeState


def create_app(state: Optional[RuntimeState] = None, dashboard_dir: str | Path = "dashboard") -> FastAPI:
    runtime = state or RuntimeState()
    app = FastAPI(title="Student Attention Realtime API", version="0.1.0")
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/realtime/latest")
    async def latest() -> Dict[str, Any]:
        with app.state.runtime.lock:
            return app.state.runtime.latest_summary or {"message": "No realtime data yet"}

    @app.get("/realtime/events")
    async def events(limit: int = 100) -> Dict[str, Any]:
        with app.state.runtime.lock:
            records = app.state.runtime.events[-max(1, min(limit, 500)) :]
        return {"count": len(records), "events": records}

    @app.post("/realtime/publish")
    async def publish(payload: Dict[str, Any]) -> JSONResponse:
        app.state.runtime.publish_sync(payload)
        return JSONResponse({"accepted": True})

    @app.websocket("/ws/realtime")
    async def ws_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        last_seen = -1
        try:
            while True:
                with app.state.runtime.lock:
                    frame_idx = app.state.runtime.latest_frame_index
                    payload = app.state.runtime.latest_summary
                if payload and frame_idx != last_seen:
                    await websocket.send_text(json.dumps(payload))
                    last_seen = frame_idx
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            pass

    dashboard = Path(dashboard_dir) / "index.html"

    @app.get("/")
    async def dashboard_index():
        if dashboard.exists():
            return FileResponse(str(dashboard))
        return JSONResponse({"message": "Dashboard not found. Expected dashboard/index.html"})

    return app

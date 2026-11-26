"""The dashboard, practice page, JSON API and audio websocket.

Served over HTTPS on port 8443 and bound to localhost, so there is no login.
Run with: ``uvicorn interviewer.web.app:create_app --factory``.
"""

from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect

from interviewer.ai.factory import build_ai
from interviewer.attendee import AttendeeClient, MeetingLinkError, status_label
from interviewer.config import Settings
from interviewer.kit import list_kits
from interviewer.session.manager import STATUS_LABELS, SessionError, SessionManager
from interviewer.session.store import SessionStore

log = logging.getLogger(__name__)

# Browsers only run module scripts served with a JavaScript type; don't rely on the OS mime table.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=HERE / "templates")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
DOWNLOADS = {"scorecard.md": "text/markdown; charset=utf-8", "transcript.txt": "text/plain; charset=utf-8"}


def build_manager(settings: Settings, http: httpx.AsyncClient) -> SessionManager:
    ai = build_ai(settings, http)
    kits, _ = list_kits(settings.kits_dir)
    attendee = (AttendeeClient(settings.attendee_base_url, settings.attendee_api_key, http)
                if settings.attendee_api_key else None)
    return SessionManager(settings, SessionStore(settings.data_dir), ai, kits, attendee, http)


def create_app(settings: Settings | None = None, manager: SessionManager | None = None, *,
               background: bool = True) -> FastAPI:
    owned_http: httpx.AsyncClient | None = None
    if manager is None:
        settings = settings or Settings.from_env()
        owned_http = httpx.AsyncClient()
        manager = build_manager(settings, owned_http)
    settings = settings or manager.settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if background:
            await manager.start_background()
        try:
            yield
        finally:
            await manager.stop_background()
            if owned_http is not None:
                await owned_http.aclose()

    app = FastAPI(title="AI Interviewer", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = manager
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def reload_kits() -> dict[str, str]:
        """Kits are re-read from KITS_DIR on each form, so edits show up without a restart."""
        kits, errors = list_kits(settings.kits_dir)
        manager.kits.clear()
        manager.kits.update(kits)
        return errors

    def kit_choices() -> list[tuple[str, str]]:
        return [(kit_id, kit.role) for kit_id, kit in sorted(manager.kits.items())]

    def page(request: Request, name: str, context: dict, status_code: int = 200) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    def not_found(request: Request) -> HTMLResponse:
        return page(request, "message.html", {"title": "Not found", "message": "There's no such session."}, 404)

    def start_page(request: Request, form: dict, error: str | None, status_code: int = 200) -> HTMLResponse:
        return page(request, "start.html", {"form": form, "error": error, "kits": kit_choices(),
                                            "attendee_ready": manager.attendee is not None}, status_code)

    @app.middleware("http")
    async def same_site_only(request: Request, call_next):
        """No login, so refuse form posts and API calls that come from another website."""
        if request.method not in SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and urlparse(origin).netloc != request.headers.get("host", ""):
                return PlainTextResponse("Blocked a request that came from another website.", status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        errors = reload_kits()
        sessions = [{**vars(record), "status_label": STATUS_LABELS.get(record.status, record.status),
                     "bot_label": status_label(record.bot_state) if record.bot_state else ""}
                    for record in manager.store.list()]
        return page(request, "dashboard.html", {"sessions": sessions, "kit_errors": errors,
                                                "attendee_ready": manager.attendee is not None,
                                                "retention_days": settings.retention_days})

    @app.get("/start", response_class=HTMLResponse)
    async def start_form(request: Request):
        reload_kits()
        return start_page(request, {}, None)

    @app.post("/sessions")
    async def create_session(request: Request, kit_id: str = Form(""), candidate_name: str = Form(""),
                             meeting_link: str = Form(""), notes: str = Form("")):
        reload_kits()
        form = {"kit_id": kit_id, "candidate_name": candidate_name, "meeting_link": meeting_link, "notes": notes}
        try:
            record = await manager.create_interview(kit_id, candidate_name, meeting_link, notes)
        except (SessionError, MeetingLinkError) as exc:
            return start_page(request, form, str(exc), 400)
        return RedirectResponse(f"/sessions/{record.id}", status_code=303)

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_page(request: Request, session_id: str):
        snapshot = manager.snapshot(session_id)
        if snapshot is None:
            return not_found(request)
        return page(request, "session.html", {"s": snapshot})

    @app.get("/sessions/{session_id}/{name}")
    async def download(request: Request, session_id: str, name: str):
        record = manager.record(session_id)
        path = manager.store.dir(record.id) / name if record is not None and name in DOWNLOADS else None
        if path is None or not path.exists():
            return PlainTextResponse("Not available (yet).", status_code=404)
        return FileResponse(path, media_type=DOWNLOADS[name], filename=f"{Path(name).stem}-{record.id}{path.suffix}")

    @app.get("/api/sessions/{session_id}")
    async def session_json(session_id: str):
        snapshot = manager.snapshot(session_id)
        if snapshot is None:
            return JSONResponse({"error": "There's no such session."}, status_code=404)
        return snapshot

    @app.post("/api/sessions/{session_id}/begin")
    async def begin(session_id: str):
        if manager.record(session_id) is None:
            return JSONResponse({"error": "There's no such session."}, status_code=404)
        try:
            await manager.begin(session_id)
        except SessionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/end")
    async def end(session_id: str):
        if manager.record(session_id) is None:
            return JSONResponse({"error": "There's no such session."}, status_code=404)
        await manager.end(session_id)
        return {"ok": True}

    @app.get("/practice", response_class=HTMLResponse)
    async def practice(request: Request, kit_id: str = ""):
        errors = reload_kits()
        if not manager.kits:
            message = "There are no usable interview kits. Fix the kit problems shown on the dashboard."
            return page(request, "message.html", {"title": "Practice", "message": message,
                                                  "details": errors}, 503)
        kit_id = kit_id if kit_id in manager.kits else sorted(manager.kits)[0]
        record = await manager.create_practice(kit_id)
        base = str(request.url_for("audio_ws", session_id=record.id))
        ws_url = base.replace("http", "ws", 1) + f"?token={record.token}"
        return page(request, "practice.html", {"record": record, "ws_url": ws_url, "kits": kit_choices(),
                                               "kit_id": kit_id})

    @app.websocket("/ws/audio/{session_id}", name="audio_ws")
    async def audio_ws(websocket: WebSocket, session_id: str, token: str = ""):
        await websocket.accept()  # accept first, so the client sees the 4403 close code

        async def send(message: dict) -> None:
            await websocket.send_json(message)

        try:
            runner = await manager.on_ws_connect(session_id, token, send)
        except PermissionError:
            await websocket.close(code=4403)
            return
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None:
                    raw = message.get("bytes")
                if raw:
                    await runner.handle_message(raw)
        except WebSocketDisconnect:
            pass
        finally:
            await manager.on_ws_disconnect(session_id, send)

    return app

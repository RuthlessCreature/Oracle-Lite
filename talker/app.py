from __future__ import annotations

import gc
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from .attachments import parse_attachment
from .config import TalkerConfig, load_config
from .history import ChatHistory
from .model import TalkerModel


MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class TalkerState:
    def __init__(self, cfg: TalkerConfig):
        self.cfg = cfg
        self.history = ChatHistory(cfg.history_db)
        self.model = TalkerModel(cfg.training_output_dir)
        viewer_url = os.environ.get("ORACLE_VIEWER_MODEL_BASE_URL", "").strip()
        self.viewer_model = (
            TalkerModel(
                cfg.training_output_dir,
                openai_base_url=viewer_url,
                openai_model=os.environ.get(
                    "ORACLE_VIEWER_MODEL", "qwen3.8-flash-next-iq2_xs"
                ),
                backend_label=os.environ.get(
                    "ORACLE_VIEWER_MODEL_BACKEND", "strata-native"
                ),
                openai_api_key=os.environ.get("ORACLE_VIEWER_MODEL_API_KEY", ""),
            )
            if viewer_url else self.model
        )
        if (viewer_url and self.model.openai_base_url.rstrip('/')==viewer_url.rstrip('/')
                and self.model.openai_model==os.environ.get('ORACLE_VIEWER_MODEL','qwen3.8-flash-next-iq2_xs')):
            self.viewer_model=self.model
        self.model_error: str | None = None
        self.viewer_model_error: str | None = None

    def load_model(self) -> None:
        try:
            self.model.load()
            self.model_error = None
        except Exception as exc:
            self.model_error = f"{type(exc).__name__}: {exc}"
        if self.viewer_model is not self.model:
            try:
                self.viewer_model.load()
                self.viewer_model_error = None
            except Exception as exc:
                self.viewer_model_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict:
        return {
            **self.model.info(),
            "error": self.model_error,
            "viewer_model": self.viewer_model.info(),
            "viewer_model_error": self.viewer_model_error,
            "training_output_dir": str(self.cfg.training_output_dir),
        }


def create_app(config_path: str | Path | None = None) -> FastAPI:
    cfg = load_config(config_path)
    state = TalkerState(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await run_in_threadpool(state.load_model)
        try:
            yield
        finally:
            app.state.viewer_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Oracle-Lite Talker",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.talker = state
    from oracle_viewer.api import attach_viewer
    attach_viewer(app, state)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        path = cfg.talker_dir / "static" / "index.html"
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/api/status")
    async def status():
        return state.status()

    @app.get("/api/conversations")
    async def conversations():
        return state.history.list_conversations()

    @app.post("/api/conversations")
    async def new_conversation():
        return state.history.new_conversation()

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        state.history.delete_conversation(conversation_id)
        shutil.rmtree(cfg.uploads_dir / conversation_id, ignore_errors=True)
        shutil.rmtree(cfg.parsed_dir / conversation_id, ignore_errors=True)
        return {"ok": True}

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(conversation_id: str):
        return state.history.get_messages(conversation_id)

    @app.post("/api/reload-model")
    async def reload_model():
        # A 9B model cannot safely coexist with its replacement on a 16GB GPU.
        # Release the old instance first, then discover/load the newest adapter.
        def reload_shared():
            # Keep the shared object/mutex stable for queued Viewer tasks.
            with state.model._lock:
                import torch
                state.model.model = None
                state.model.processor = None
                state.model.bundle = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                state.load_model()
        await run_in_threadpool(reload_shared)
        if state.model_error:
            raise HTTPException(status_code=500, detail=state.model_error)
        return state.status()

    async def save_upload(
        conversation_id: str,
        upload: UploadFile,
    ) -> Path:
        suffix = Path(upload.filename or "attachment").suffix.lower()
        directory = cfg.uploads_dir / conversation_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{uuid.uuid4().hex}{suffix}"
        written = 0
        try:
            with target.open("wb") as out:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Attachment exceeds 512 MiB: {upload.filename}",
                        )
                    out.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return target.resolve()

    @app.post("/api/chat")
    async def chat(
        conversation_id: str | None = Form(default=None),
        message: str = Form(default=""),
        project_id: str | None = Form(default=None),
        fai: int | None = Form(default=None),
        files: list[UploadFile] | None = File(default=None),
    ):
        if state.model_error or not state.model.loaded:
            raise HTTPException(
                status_code=503,
                detail=state.model_error or "Talker model is not loaded",
            )

        uploads = files or []
        text = message.strip()
        if not text and not uploads:
            raise HTTPException(status_code=400, detail="Message or attachment required")

        project_detail = None
        if project_id:
            requested = re.search(r'(?<![A-Za-z0-9_])FAI\s*0*(\d+)(?!\d)', text, re.I)
            number = int(requested.group(1)) if requested else fai
            if number is None:
                raise HTTPException(400, "项目对话需要指定 FAI 编号")
            try:
                project_detail = state.viewer.detail(project_id, number)
            except (KeyError, ValueError) as exc:
                raise HTTPException(404, str(exc)) from exc

        conversation_id = state.history.ensure_conversation(conversation_id)
        user_message_id = state.history.add_message(
            conversation_id,
            "user",
            text,
        )

        for upload in uploads:
            stored = await save_upload(conversation_id, upload)
            asset_dir = (
                cfg.parsed_dir
                / conversation_id
                / user_message_id
                / uuid.uuid4().hex
            )
            try:
                parsed_text, images = await run_in_threadpool(
                    parse_attachment,
                    stored,
                    asset_dir,
                )
            except Exception as exc:
                parsed_text = (
                    f"Attachment parsing failed for {upload.filename}: "
                    f"{type(exc).__name__}: {exc}"
                )
                images = []

            state.history.add_attachment(
                user_message_id,
                filename=upload.filename or stored.name,
                stored_path=str(stored),
                mime_type=upload.content_type,
                parsed_text=parsed_text,
                images=images,
            )

        current_messages = state.history.get_messages(conversation_id)
        if len(current_messages) == 1:
            title = text or (
                current_messages[0]["attachments"][0]["filename"]
                if current_messages[0]["attachments"]
                else "New chat"
            )
            state.history.rename_conversation(conversation_id, title[:60])

        current_messages = state.history.get_messages(conversation_id)
        try:
            if project_detail:
                from oracle_viewer.assistant import answer_project
                task = state.viewer.task(project_id, project_detail['item']['number'], text)
                state.viewer.update_task(project_id, task['id'], status='running', conversation_id=conversation_id)
                try:
                    result = await run_in_threadpool(
                        answer_project, state.model,
                        f"针对 FAI {project_detail['item']['number']}，{text}",
                        task['input_context'], task['binding'],
                    )
                    state.viewer.update_task(project_id, task['id'], status='complete', result=result)
                    answer = result['text']
                except Exception as exc:
                    state.viewer.update_task(project_id, task['id'], status='failed', error=str(exc))
                    raise
            else:
                answer = await run_in_threadpool(
                    state.model.answer,
                    current_messages,
                )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        state.history.add_message(conversation_id, "assistant", answer)
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "messages": state.history.get_messages(conversation_id),
        }

    return app

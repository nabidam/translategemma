"""
title: Wan2.2 Text-to-Video Pipe (Deterministic)
author: TranslateGemma Team
description: Select this pipe as the model and every message becomes a video-generation request. The scene description goes straight to Wan2.2-T2V-A14B on vllm-omni with no chat LLM in the loop: an async job is created, polled with live status chips, and the finished MP4 is stored in OpenWebUI file storage, attached to the chat as a downloadable file, and rendered as an inline video player. Optional deterministic flags: /video 832x480 5s steps=40 seed=42 <prompt>.
version: 1.0.0
license: MIT
requirements: requests, pydantic
"""

# Self-contained OpenWebUI pipe (Workspace > Functions) for Wan2.2 text-to-
# video on vllm-omni. OpenWebUI execs this file as a standalone module
# (backend/open_webui/utils/plugin.py), so nothing is imported from this repo.
# `requests` is a declared requirement; `open_webui.*` is touched lazily
# inside functions, which also lets the helpers be unit-tested out of process.
#
# v0.11.3 pipe mechanics this file relies on (backend/open_webui/functions.py,
# generate_function_chat_completion):
#   - pipe() is called with `body` (the chat completion body) plus exactly the
#     dunders declared in its signature: __user__ (with UserValves),
#     __chat_id__, __message_id__, __files__, __event_emitter__, __request__...
#   - the return value may be a str (sent as the assistant reply, in stream or
#     non-stream mode), a text generator (streamed delta by delta), a dict, a
#     BaseModel, or a StreamingResponse. It must NOT be an HTMLResponse:
#     process_tool_result-style embed handling is tool-only.
#   - inline UI therefore goes through the SOCKET event channel: emitting
#     {"type": "chat:message:embeds", "data": {"embeds": [html]}} makes the
#     frontend set message.embeds, which ResponseMessage renders in a
#     FullHeightIframe (srcdoc sandbox, popups allowed). The tool path emits
#     the same event family via process_tool_result; the pipe emits it
#     directly.
#   - a pipe has no execution timeout (the caller awaits it bare); the only
#     budgets are TOOL_TIMEOUT here and the reverse proxy on the chat stream.

import asyncio
import io
import re
import time
import uuid
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel, Field

# ------------------------------------------------------------------- player

_PLAYER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;background:transparent}}
  .wrap{{display:flex;flex-direction:column;gap:6px;padding:4px}}
  video{{width:100%;max-height:440px;border-radius:12px;background:#000;display:block}}
  .meta{{font:12px/1.4 system-ui,sans-serif;color:#888;display:flex;
         justify-content:space-between;align-items:center;gap:8px}}
  a{{color:#888;text-decoration:none}}
  a:hover{{text-decoration:underline}}
</style></head><body>
<div class="wrap">
  <video src="{src}" controls playsinline preload="metadata"></video>
  <div class="meta"><span>{name}</span>
    <a href="{src}" target="_blank" rel="noopener">open in new tab</a></div>
</div></body></html>
"""

# The T5 text encoder of the Wan checkpoints truncates very long prompts
# silently; fail loudly instead of rendering a half-prompt. No LLM is in the
# loop to summarize, so the deterministic answer is "shorten it".
_MAX_PROMPT_CHARS = 2000

_REFERENTIAL = {
    "",
    "this",
    "that",
    "it",
    "these",
    "the above",
    "the above one",
    "the above video",
    "previous",
    "previous video",
    "last one",
    "last video",
    "again",
    "again please",
    "once more",
    "another",
    "make another",
    "make another one",
    "regenerate",
    "redo",
    "try again",
    "دوباره",
}

# /video flag tokens. Everything that is not consumed as a flag (and everything
# after the first line) is the prompt — deterministic, no LLM parsing.
_FLAG_RES = (
    (re.compile(r"^(\d{2,5})x(\d{2,5})$", re.IGNORECASE), "size"),
    (re.compile(r"^(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?$", re.IGNORECASE), "seconds"),
    (re.compile(r"^frames=(\d+)$", re.IGNORECASE), "frames"),
    (re.compile(r"^steps=(\d+)$", re.IGNORECASE), "steps"),
    (re.compile(r"^seed=(\d+)$", re.IGNORECASE), "seed"),
    (re.compile(r"^fps=(\d+)$", re.IGNORECASE), "fps"),
    (re.compile(r"^cfg=([0-9.]+)$", re.IGNORECASE), "cfg"),
)


# ------------------------------------------------------------- message utils


def msg_text(message: dict) -> str:
    """Message content as text, for both plain strings and content-part lists."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def last_user_message(messages) -> tuple:
    """(index, text) of the most recent user message, or (-1, '')."""
    if not messages:
        return -1, ""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return i, msg_text(msg)
    return -1, ""


def strip_wrappers(text: str) -> str:
    """Remove OpenWebUI tag wrappers that ride along with the message text."""
    text = re.sub(r"<attached_files>.*?</attached_files>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<knowledge>.*?</knowledge>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<context>.*?</context>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_video_command(text: str) -> tuple:
    """Split an optional '/video [flags] prompt' into (flags, prompt).

    Without the /video prefix the whole message is the prompt. Flags are
    space-separated tokens on the first line: 832x480, 5s, frames=49,
    steps=40, seed=42, fps=24, cfg=4.0. The first non-flag token ends the
    flag scan; the remainder of the line plus all following lines is the
    prompt.
    """
    clean = (text or "").strip()
    m = re.match(r"^/video[\s:]+(.+)$", clean, re.DOTALL | re.IGNORECASE)
    if not m:
        return {}, clean
    body = m.group(1).strip()
    first_line, _, rest = body.partition("\n")
    tokens = first_line.split()
    flags = {}
    i = 0
    while i < len(tokens):
        matched = False
        for rx, key in _FLAG_RES:
            mm = rx.match(tokens[i])
            if mm:
                flags[key] = mm.groups() if key == "size" else mm.group(1)
                i += 1
                matched = True
                break
        if not matched:
            break
    head = " ".join(tokens[i:]).strip()
    prompt = (head + ("\n" + rest.strip() if rest.strip() else "")).strip()
    return flags, prompt


def earlier_user_prompt(messages, current_idx: int) -> str:
    """Most recent user message BEFORE `current_idx`, wrappers stripped.

    The deterministic target for referential messages ("again", "this"):
    regenerate the previous scene. Returns '' when there is none.
    """
    if not messages or current_idx < 0:
        return ""
    for i in range(current_idx - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return strip_wrappers(msg_text(msg))
    return ""


# --------------------------------------------------------------- video job


class Wan22VideoClient:
    """Thin async client for the vllm-omni /v1/videos job API."""

    def __init__(self, base_url: str, request_timeout: int):
        self.base = base_url.rstrip("/")
        self.request_timeout = request_timeout

    @staticmethod
    def _multipart(form: dict) -> dict:
        """Encode every field as multipart (matches the documented -F calls)."""
        return {k: (None, str(v)) for k, v in form.items() if v is not None}

    def submit(self, form: dict) -> dict:
        resp = requests.post(
            f"{self.base}/v1/videos",
            data=self._multipart(form),
            headers={"Accept": "application/json"},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"vllm-omni rejected the job (HTTP {resp.status_code}): {resp.text[:500]}")
        try:
            return resp.json()
        except Exception:
            raise RuntimeError(f"unexpected non-JSON response from vllm-omni: {resp.text[:500]}")

    def status(self, video_id: str) -> dict:
        resp = requests.get(f"{self.base}/v1/videos/{video_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download(self, video_id: str) -> tuple:
        resp = requests.get(
            f"{self.base}/v1/videos/{video_id}/content",
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", "video/mp4")

    def delete(self, video_id: str) -> None:
        requests.delete(f"{self.base}/v1/videos/{video_id}", timeout=30)


# ---------------------------------------------------------------- delivery


async def upload_video_file(video_bytes: bytes, name: str, content_type: str, user_id: str):
    """Store the MP4 in OpenWebUI file storage (no RAG processing).

    Two-step pattern of OpenWebUI's own upload_image helper:
    Storage.upload_file for the bytes, Files.insert_new_file for the DB
    record. Returns the FileModel record, or None when storage is
    unavailable (the caller then reports the server-side job id instead).
    """
    try:
        from open_webui.models.files import FileForm, Files
        from open_webui.storage.provider import Storage
    except Exception:
        return None
    try:
        file_id = str(uuid.uuid4())
        contents, file_path = await asyncio.to_thread(
            Storage.upload_file,
            io.BytesIO(video_bytes),
            f"{file_id}_{name}",
            {
                "OpenWebUI-User-Id": user_id,
                "OpenWebUI-File-Id": file_id,
            },
        )
        return await Files.insert_new_file(
            user_id,
            FileForm(
                id=file_id,
                filename=name,
                path=file_path,
                # No 'content': binary. status=completed keeps the extraction
                # pipeline away from the MP4.
                data={"status": "completed"},
                meta={
                    "name": name,
                    "content_type": content_type,
                    "size": len(contents),
                    "data": {},
                },
            ),
        )
    except Exception:
        return None


async def deliver_video(
    record,
    name: str,
    content_type: str,
    size: int,
    embed_player: bool,
    __chat_id__: Optional[str],
    __message_id__: Optional[str],
    __event_emitter__: Optional[Callable[[dict], Any]],
) -> None:
    """Attach the MP4 to the chat: file chip (persisted) + inline player.

    Chip path (always, when a record exists): persist to the message via
    Chats.add_message_files_by_id_and_message_id, then push it live via the
    chat:message:files socket event — the same mechanism the TranslateGemma
    tool uses, proven on this deployment.

    Player path (EMBED_PLAYER): emit chat:message:embeds with a small HTML
    document whose <video> points at /files/{id}/content. The frontend
    (Chat.svelte) sets message.embeds on that event and ResponseMessage
    renders it in a sandboxed FullHeightIframe. A srcdoc iframe inherits the
    parent page as base URL, and media subresource loads need no CORS, so
    the same-origin file URL plays even without allow-same-origin. The
    embed survives reload: the frontend saves message.embeds with the chat.
    """
    file_entry = {
        "type": "file",
        "name": name,
        "content_type": content_type,
        "size": size,
        "id": record.id,
        "url": record.id,
        "file": record.model_dump(),
    }
    chat_id = str(__chat_id__ or "")
    if chat_id and __message_id__ and not chat_id.startswith(("temp:", "channel:")):
        try:
            from open_webui.models.chats import Chats

            saved = await Chats.add_message_files_by_id_and_message_id(
                chat_id, str(__message_id__), [file_entry]
            )
            if saved:
                file_entry = saved[0]
        except Exception:
            pass
    if __event_emitter__:
        try:
            await __event_emitter__(
                {"type": "chat:message:files", "data": {"files": [file_entry]}}
            )
        except Exception:
            pass
    if embed_player and __event_emitter__:
        try:
            html = _PLAYER_HTML.format(src=f"/files/{record.id}/content", name=name)
            await __event_emitter__(
                {"type": "chat:message:embeds", "data": {"embeds": [html]}}
            )
        except Exception:
            pass


# --------------------------------------------------------------------- pipe


class Pipe:
    class Valves(BaseModel):
        VLLM_OMNI_BASE_URL: str = Field(
            default="http://vllm-omni:8091",
            description="Base URL of the vllm-omni server as reachable from the OpenWebUI backend (no trailing slash)",
        )
        WIDTH: int = Field(default=832, description="Default output width (px)")
        HEIGHT: int = Field(default=480, description="Default output height (px)")
        NUM_FRAMES: int = Field(default=33, description="Default frame count (~2s at 16 fps)")
        FPS: int = Field(default=16, description="Default output fps")
        NUM_INFERENCE_STEPS: int = Field(default=40, description="Default denoising steps")
        GUIDANCE_SCALE: float = Field(default=4.0, description="CFG scale, low-noise stage")
        GUIDANCE_SCALE_2: float = Field(default=4.0, description="CFG scale, high-noise stage (Wan2.2)")
        BOUNDARY_RATIO: float = Field(default=0.875, description="Low/high DiT boundary split (Wan2.2)")
        FLOW_SHIFT: float = Field(default=5.0, description="Scheduler flow shift (Wan2.2)")
        NEGATIVE_PROMPT: str = Field(
            default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            description="Default negative prompt (vllm-omni T2V example); leave blank to disable",
        )
        POLL_INTERVAL: int = Field(default=5, description="Seconds between job status polls")
        TOOL_TIMEOUT: int = Field(
            default=1800,
            description="Overall wall-clock budget for one generation, seconds",
        )
        REQUEST_TIMEOUT: int = Field(
            default=300,
            description="Per-request timeout to vllm-omni (the MP4 download can be large)",
        )
        EMBED_PLAYER: bool = Field(
            default=True,
            description="Render an inline <video> player in the chat (HTML embed). The file chip is always attached.",
        )
        ATTACH_FILE: bool = Field(
            default=True,
            description="Attach the MP4 as a downloadable chat file (stored in OpenWebUI file storage).",
        )
        CLEANUP_JOB: bool = Field(
            default=True,
            description="DELETE the vllm-omni job after successful upload (the MP4 then lives in OpenWebUI storage)",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _status(
        self,
        __event_emitter__: Optional[Callable[[dict], Any]],
        description: str,
        done: bool = False,
    ):
        if not __event_emitter__:
            return
        try:
            await __event_emitter__(
                {"type": "status", "data": {"description": description, "done": done}}
            )
        except Exception:
            pass

    def _build_form(self, flags: dict, prompt: str) -> dict:
        """Valve defaults; explicit /video flags win."""
        v = self.valves
        form: dict = {"prompt": prompt}
        size = flags.get("size")
        if size:
            form["size"] = f"{size[0]}x{size[1]}"
        else:
            form["width"] = v.WIDTH
            form["height"] = v.HEIGHT
        if flags.get("seconds"):
            form["seconds"] = flags["seconds"]
        elif flags.get("frames"):
            form["num_frames"] = int(flags["frames"])
        else:
            form["num_frames"] = v.NUM_FRAMES
        form["fps"] = int(flags.get("fps") or v.FPS)
        form["num_inference_steps"] = int(flags.get("steps") or v.NUM_INFERENCE_STEPS)
        form["guidance_scale"] = float(flags.get("cfg") or v.GUIDANCE_SCALE)
        form["guidance_scale_2"] = v.GUIDANCE_SCALE_2
        form["boundary_ratio"] = v.BOUNDARY_RATIO
        form["flow_shift"] = v.FLOW_SHIFT
        if flags.get("seed") is not None:
            form["seed"] = int(flags["seed"])
        neg = v.NEGATIVE_PROMPT.strip()
        if neg:
            form["negative_prompt"] = neg
        return form

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ):
        """Handle one chat completion. The message IS the video request;
        no chat LLM is involved at any point. Returns the reply text
        (stream mode wraps it in a single delta chunk)."""
        body = body or {}
        messages = body.get("messages") or []

        current_idx, current_text = last_user_message(messages)
        flags, prompt = parse_video_command(strip_wrappers(current_text))

        # Deterministic reference resolution: "again" / "this" reuses the
        # previous scene (fresh seed). A pipe has no LLM to interpret
        # partial modifications, so only exact referentials fall back.
        if prompt.strip().lower() in _REFERENTIAL:
            prompt = earlier_user_prompt(messages, current_idx)
            flags = {}
        prompt = prompt.strip()

        if not prompt:
            return (
                "Nothing to generate. Every message sent to this pipe is a "
                "video prompt — describe the scene you want. Optional flags "
                "after /video: 832x480 (size), 5s (duration), frames=49, "
                "steps=40, seed=42, fps=24, cfg=4.0. Example: /video 832x480 "
                "5s A snow leopard walks through morning fog."
            )
        if len(prompt) > _MAX_PROMPT_CHARS:
            return (
                f"Prompt too long ({len(prompt)} chars; limit {_MAX_PROMPT_CHARS}). "
                "The Wan text encoder truncates silently — shorten the "
                "description and resend."
            )

        v = self.valves
        client = Wan22VideoClient(v.VLLM_OMNI_BASE_URL, v.REQUEST_TIMEOUT)
        form = self._build_form(flags, prompt)
        width, height = (
            (form["size"].split("x")) if "size" in form else (form["width"], form["height"])
        )

        # ---- submit --------------------------------------------------------
        await self._status(__event_emitter__, "Submitting video job to vllm-omni…")
        try:
            job = await asyncio.to_thread(client.submit, form)
        except Exception as error:
            return (
                f"Error: could not submit the video job: {error}. Check the "
                "VLLM_OMNI_BASE_URL valve and the network path from the "
                "OpenWebUI backend to the model server."
            )
        video_id = job.get("id")
        if not video_id:
            return f"Error: vllm-omni returned no job id: {str(job)[:500]}"

        # ---- poll ----------------------------------------------------------
        deadline = time.monotonic() + v.TOOL_TIMEOUT
        started = time.monotonic()
        poll_failures = 0
        status = "queued"
        while True:
            if time.monotonic() > deadline:
                return (
                    f"Error: the video job ({video_id}) did not finish within "
                    f"{v.TOOL_TIMEOUT // 60} minutes. It may still be running "
                    "server-side; an admin can list jobs with GET /v1/videos "
                    f"on {v.VLLM_OMNI_BASE_URL} and download the MP4 from "
                    f"/v1/videos/{video_id}/content."
                )
            await asyncio.sleep(v.POLL_INTERVAL)
            elapsed = int(time.monotonic() - started)
            try:
                job = await asyncio.to_thread(client.status, video_id)
                poll_failures = 0
            except Exception:
                poll_failures += 1
                if poll_failures >= 6:
                    return (
                        f"Error: lost contact with the vllm-omni server while "
                        f"polling job {video_id} (6 consecutive failures). The "
                        "job may still be running server-side."
                    )
                await self._status(
                    __event_emitter__,
                    f"Video job {video_id}: {status} ({elapsed}s) — polling hiccup, retrying…",
                )
                continue
            status = str(job.get("status") or status)
            await self._status(
                __event_emitter__, f"Video job {video_id}: {status} ({elapsed}s)…"
            )
            if status == "completed":
                break
            if status == "failed":
                detail = {
                    k: job.get(k)
                    for k in ("error", "failure_reason", "detail", "message")
                    if job.get(k)
                }
                return (
                    "Error: video generation failed on the server. "
                    f"Job record: {str(detail or job)[:500]}"
                )
            # queued / in_progress / anything unknown: keep polling

        # ---- download -------------------------------------------------------
        await self._status(
            __event_emitter__, f"Video job {video_id} completed — downloading MP4…"
        )
        try:
            video_bytes, content_type = await asyncio.to_thread(client.download, video_id)
        except Exception as error:
            return f"Error: the job completed but the video download failed: {error}"
        if not video_bytes:
            return "Error: the server returned an empty video file."

        # ---- store + deliver ------------------------------------------------
        user_id = str((__user__ or {}).get("id") or "")
        name = f"wan22-{int(time.time())}.mp4"
        record = await upload_video_file(video_bytes, name, content_type, user_id)
        if record is None:
            return (
                "Error: the video was generated but OpenWebUI file storage was "
                f"unavailable, so it could not be attached. Job id {video_id} "
                f"is still downloadable from {v.VLLM_OMNI_BASE_URL}/v1/videos/"
                f"{video_id}/content on the prod machine."
            )

        await self._status(__event_emitter__, "Attaching video to the chat…")
        if v.ATTACH_FILE:
            await deliver_video(
                record,
                name,
                content_type,
                len(video_bytes),
                v.EMBED_PLAYER,
                __chat_id__,
                __message_id__,
                __event_emitter__,
            )

        if v.CLEANUP_JOB:
            try:
                await asyncio.to_thread(client.delete, video_id)
            except Exception:
                pass

        elapsed = int(time.monotonic() - started)
        await self._status(
            __event_emitter__,
            f"Video ready in {elapsed}s ({len(video_bytes) // 1024} KB).",
            done=True,
        )

        # A plain str is a valid pipe return in BOTH stream and non-stream
        # mode (stream wraps it in one delta chunk + finish), so no
        # chunker is needed for a short reply.
        return (
            f"Video ready — {width}x{height}, {form.get('num_frames', '?')} frames "
            f"@ {form.get('fps', '?')} fps, {elapsed}s on the GPU. It is playing "
            "in the chat; use the file chip to download the MP4."
        )

"""Async client for OpenAI-compatible endpoints: vLLM, llama-server, or a hosted API (OpenAI, Gemini, Claude, ...)."""

import asyncio
import base64
import io
import os
import re
import time
from dataclasses import asdict, dataclass

from PIL import Image

MAX_PIXELS = 1536 * 1536
MAX_SIDE = 2048


@dataclass
class Prediction:
    text: str
    reasoning: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    sent_size: list[int] | None = None  # [w, h] of the first image as sent
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def encode_image(path: str, max_pixels: int = MAX_PIXELS) -> tuple[str, list[int]]:
    im = Image.open(path)
    im = im.convert("RGB") if im.mode != "RGB" else im
    w, h = im.size
    scale = min(1.0, (max_pixels / (w * h)) ** 0.5, MAX_SIDE / max(w, h))
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), list(im.size)


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class FatalAPIError(RuntimeError):
    """The endpoint refuses every request (bad key, no access, unknown model): stop the run instead of recording an
    error for each item."""


def api_key_from_env(var: str | None) -> str:
    """The API key held in environment variable `var` (the key itself is never passed on the command line or saved)."""
    if not var:
        return "EMPTY"
    key = os.environ.get(var)
    if not key:
        raise FatalAPIError(f"environment variable {var} is not set (it should hold the API key)")
    return key


def api_base(url: str) -> str:
    """OpenAI-style base URL: a bare server address gets /v1; a URL that already has a version path is kept."""
    url = url.rstrip("/")
    return url if re.search(r"/v\d+[a-z0-9]*(/|$)", url) else url + "/v1"


class Client:
    def __init__(self, base_url: str, model: str | None = None, api_key: str = "EMPTY", timeout: float = 1800,
                 max_pixels: int = MAX_PIXELS, retries: int = 3, prefer: str | None = None):
        """`model`: the name to send. Unset: `prefer` if the endpoint lists it (or does not list models), else the
        endpoint's only (or first) model."""
        from openai import AsyncOpenAI

        self.api = AsyncOpenAI(base_url=api_base(base_url), api_key=api_key, timeout=timeout, max_retries=0)
        self.model, self.prefer, self.max_pixels, self.retries = model, prefer, max_pixels, retries
        self.token_param, self.dropped = "max_tokens", set()  # adapted to what the endpoint accepts
        self._lock = asyncio.Lock()

    async def resolve_model(self) -> str:
        async with self._lock:
            if not self.model:
                try:
                    ids = [m.id for m in (await self.api.models.list()).data]
                except Exception as e:
                    if _fatal(e) and not _status(e) == 404:
                        raise FatalAPIError(_describe(e)) from e
                    ids = []
                if self.prefer and (self.prefer in ids or not ids or len(ids) > 1):
                    self.model = self.prefer
                elif ids:
                    self.model = ids[0]
                else:
                    raise FatalAPIError("the endpoint lists no models; pass the model name")
        return self.model

    def _adapt(self, e: Exception) -> bool:
        """Switch to the parameter spelling a hosted API asks for (e.g. max_completion_tokens; no temperature)."""
        msg = str(e).lower()
        if _status(e) != 400:
            return False
        if "max_tokens" in msg and "max_completion_tokens" in msg and self.token_param == "max_tokens":
            self.token_param = "max_completion_tokens"
            return True
        for k in ("temperature", "top_p", "top_k", "seed"):
            if k in msg and ("unsupported" in msg or "not support" in msg or "unrecognized" in msg or "unknown" in msg) \
                    and k not in self.dropped:
                self.dropped.add(k)
                return True
        return False

    async def chat(self, prompt: str, images: list[str], system: str | None = None, max_tokens: int = 16384,
                   body: dict | None = None) -> Prediction:
        """One request. `body` holds sampling + chat_template_kwargs etc. (sent as extra_body)."""
        model = await self.resolve_model()
        encoded = [encode_image(p, self.max_pixels) for p in images]
        content = [{"type": "image_url", "image_url": {"url": u}} for u, _ in encoded] + [{"type": "text", "text": prompt}]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        t0, err, attempt = time.perf_counter(), None, 0
        while attempt <= self.retries:
            b = {k: v for k, v in (body or {}).items() if k not in self.dropped}
            kw = {k: b.pop(k) for k in ("temperature", "top_p", "seed") if k in b}
            kw[self.token_param] = max_tokens
            try:
                r = await self.api.chat.completions.create(model=model, messages=messages, extra_body=b or None, **kw)
                msg = r.choices[0].message
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                usage = r.usage
                return Prediction(text=msg.content or "", reasoning=reasoning, finish_reason=r.choices[0].finish_reason or "",
                                  prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                                  completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                                  latency_s=time.perf_counter() - t0, sent_size=encoded[0][1] if encoded else None)
            except Exception as e:  # connection drops, 5xx, timeouts, rate limits
                if self._adapt(e):
                    continue
                if _fatal(e):
                    raise FatalAPIError(_describe(e)) from e
                err = f"{type(e).__name__}: {e}"
                if _status(e) == 400:  # this request is refused as it is (e.g. image too large); retrying will not help
                    break
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt * (20 if _status(e) == 429 else 5))
                attempt += 1
        return Prediction(text="", finish_reason="error", latency_s=time.perf_counter() - t0,
                          sent_size=encoded[0][1] if encoded else None, error=err)


def _status(e: Exception) -> int | None:
    return getattr(e, "status_code", None)


def _fatal(e: Exception) -> bool:
    return _status(e) in (401, 403, 404)


def _describe(e: Exception) -> str:
    what = {401: "API key rejected (401)", 403: "no access to this model (403)", 404: "model or endpoint not found (404)"}
    return f"{what.get(_status(e), type(e).__name__)}: {str(e)[:300]}"

"""Async client for OpenAI-compatible VLM servers (vLLM, llama-server)."""

import asyncio
import base64
import io
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


class Client:
    def __init__(self, base_url: str, model: str | None = None, api_key: str = "EMPTY", timeout: float = 1800,
                 max_pixels: int = MAX_PIXELS, retries: int = 3):
        from openai import AsyncOpenAI

        self.api = AsyncOpenAI(base_url=base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/v1") else "/v1"),
                               api_key=api_key, timeout=timeout, max_retries=0)
        self.model, self.max_pixels, self.retries = model, max_pixels, retries

    async def resolve_model(self) -> str:
        if not self.model:
            self.model = (await self.api.models.list()).data[0].id
        return self.model

    async def chat(self, prompt: str, images: list[str], system: str | None = None, max_tokens: int = 16384,
                   body: dict | None = None) -> Prediction:
        """One request. `body` holds sampling + chat_template_kwargs etc. (sent as extra_body)."""
        model = await self.resolve_model()
        encoded = [encode_image(p, self.max_pixels) for p in images]
        content = [{"type": "image_url", "image_url": {"url": u}} for u, _ in encoded] + [{"type": "text", "text": prompt}]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = dict(body or {})
        temperature = body.pop("temperature", None)
        t0, err = time.perf_counter(), None
        for attempt in range(self.retries + 1):
            try:
                r = await self.api.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                           temperature=temperature, extra_body=body or None)
                msg = r.choices[0].message
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                usage = r.usage
                return Prediction(text=msg.content or "", reasoning=reasoning, finish_reason=r.choices[0].finish_reason or "",
                                  prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                                  completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                                  latency_s=time.perf_counter() - t0, sent_size=encoded[0][1] if encoded else None)
            except Exception as e:  # connection drops, 5xx, timeouts
                err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt * 5)
        return Prediction(text="", finish_reason="error", latency_s=time.perf_counter() - t0,
                          sent_size=encoded[0][1] if encoded else None, error=err)

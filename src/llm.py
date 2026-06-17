"""LLM provider abstraction.

Generation is the only part of the system that needs an external API. We keep it
behind a small interface so the provider can be swapped (Gemini by default). The
rest of the codebase only ever calls `get_llm().generate(...)`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from .config import CONFIG


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """Raised when every candidate model is out of free-tier daily quota."""


class _RateLimiter:
    """Process-wide minimum spacing between LLM calls (free tier ~15 RPM)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


_RATE = _RateLimiter(CONFIG.llm_min_interval)


def _retry_delay_seconds(err_text: str) -> float | None:
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s", err_text)
    return float(m.group(1)) if m else None


class LLMProvider:
    """Interface every provider implements."""

    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, system: str | None = None,
                      temperature: float = 0.0) -> Any:
        """Generate and parse a JSON object/array, tolerating code fences."""
        raw = self.generate(prompt, system=system, temperature=temperature)
        return _extract_json(raw)


# --------------------------------------------------------------------------- #
# Gemini (default)                                                            #
# --------------------------------------------------------------------------- #
class GeminiProvider(LLMProvider):
    # Tried in order. flash-lite first (largest free-tier daily quota); only valid,
    # currently-served model names (no dead gemini-1.5-flash).
    FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]

    def __init__(self, model: str | None = None, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError(
                "No Gemini API key found. Set GEMINI_API_KEY in your environment "
                "or .env file. Get a free key at https://aistudio.google.com/app/apikey"
            )
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise LLMError("google-genai not installed. Run: pip install google-genai") from e

        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)
        preferred = model or CONFIG.llm_model
        self.models = [preferred] + [m for m in self.FALLBACK_MODELS if m != preferred]
        self._dead: set[str] = set()  # models known to be out of daily quota

    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> str:
        kwargs = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # 2.5 models "think" by default; disable it for our short structured tasks
        # to cut latency and token usage. Ignored gracefully on models without it.
        try:
            kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        cfg = self._types.GenerateContentConfig(**kwargs)
        last_err: Exception | None = None
        live = [m for m in self.models if m not in self._dead] or self.models
        for model in live:
            attempt = 0
            while attempt < 4:
                _RATE.wait()
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=prompt, config=cfg
                    )
                    text = (resp.text or "").strip()
                    if text:
                        return text
                    last_err = LLMError("empty response")
                    attempt += 1
                    continue
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    low = msg.lower()
                    is_quota = "429" in msg or "resource_exhausted" in low or "quota" in low
                    if is_quota:
                        delay = _retry_delay_seconds(msg)
                        if delay is not None and delay <= 90:
                            # recoverable per-minute cap: wait it out, retry SAME model
                            time.sleep(delay + 2)
                            attempt += 1
                            continue
                        self._dead.add(model)  # daily cap -> skip from now on
                        break
                    if "503" in msg or "unavailable" in low:
                        time.sleep(2 * (attempt + 1))
                        attempt += 1
                        continue
                    break  # non-transient -> next model
        if self._dead and len(self._dead) >= len(self.models):
            raise QuotaExhausted(
                "All Gemini models are out of free-tier DAILY quota. "
                "Wait for the daily reset, switch GEMINI_MODEL, or use another key. "
                f"(last error: {str(last_err)[:160]})"
            )
        raise LLMError(f"Gemini generation failed: {last_err}")


# --------------------------------------------------------------------------- #
# Groq (OpenAI-compatible; fast, generous free tier)                          #
# --------------------------------------------------------------------------- #
class GroqProvider(LLMProvider):
    """Talks to Groq's OpenAI-compatible chat endpoint over stdlib HTTP.

    No extra dependency required. Set GROQ_API_KEY (and optionally GROQ_MODEL).
    """
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    def __init__(self, model: str | None = None, api_key: str | None = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMError(
                "No Groq API key found. Set GROQ_API_KEY in your environment or "
                ".env file. Get a free key at https://console.groq.com/keys"
            )
        self.api_key = api_key
        preferred = model or os.getenv("GROQ_MODEL") or CONFIG.groq_model
        self.models = [preferred] + [m for m in self.FALLBACK_MODELS if m != preferred]
        self._dead: set[str] = set()

    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> str:
        import urllib.error
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        live = [m for m in self.models if m not in self._dead] or self.models
        for model in live:
            attempt = 0
            while attempt < 4:
                _RATE.wait()
                body = json.dumps({
                    "model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens,
                }).encode()
                req = urllib.request.Request(
                    self.ENDPOINT, data=body, method="POST",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json",
                             "User-Agent": "advanced-rag-icog/1.0"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read().decode())
                    text = (data["choices"][0]["message"]["content"] or "").strip()
                    if text:
                        return text
                    last_err = LLMError("empty response")
                    attempt += 1
                    continue
                except urllib.error.HTTPError as e:
                    detail = e.read().decode(errors="replace")
                    last_err = LLMError(f"Groq HTTP {e.code}: {detail[:200]}")
                    if e.code == 429:  # rate limit
                        delay = _retry_delay_seconds(detail) or float(
                            e.headers.get("retry-after", 0) or 0)
                        if delay and delay <= 90:
                            time.sleep(delay + 1)
                            attempt += 1
                            continue
                        self._dead.add(model)
                        break
                    if e.code in (500, 502, 503):
                        time.sleep(2 * (attempt + 1))
                        attempt += 1
                        continue
                    break  # 4xx other than 429 -> try next model
                except Exception as e:
                    last_err = e
                    attempt += 1
                    time.sleep(1)
        if self._dead and len(self._dead) >= len(self.models):
            raise QuotaExhausted(f"All Groq models rate/quota limited. (last: {str(last_err)[:160]})")
        raise LLMError(f"Groq generation failed: {last_err}")


_PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}
_LLM_SINGLETON: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Lazily build (and cache) the configured provider."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        provider = CONFIG.llm_provider.lower()
        if provider not in _PROVIDERS:
            raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Available: {list(_PROVIDERS)}")
        _LLM_SINGLETON = _PROVIDERS[provider]()
    return _LLM_SINGLETON


def _extract_json(raw: str) -> Any:
    """Pull a JSON value out of an LLM response, tolerating ```json fences."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...} or [...]
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = text.find(opener), text.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    continue
        raise LLMError(f"Could not parse JSON from LLM response: {raw[:200]!r}")

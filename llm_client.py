import hashlib
import json
import os
import random
import threading
import time
import zipfile

import requests

import config
from common import ensure_dirs

_price_lock = threading.Lock()
_prices = None


class LLMError(RuntimeError):
    pass


def _cache_path(key: str):
    d = config.LLM_CACHE / key[:2]
    ensure_dirs(d)
    return d / f"{key}.json"


def _prices_per_token():
    
    global _prices
    with _price_lock:
        if _prices is not None:
            return _prices
        p = config.LLM_CACHE / "models_catalogue.json"
        try:
            archive = config.ROOT / "cache" / "llm_cache.zip"
            if (not p.exists() and not archive.is_file()) or (p.exists() and time.time() - p.stat().st_mtime > 7 * 86400):
                ensure_dirs(config.LLM_CACHE)
                r = requests.get(config.OPENROUTER_MODELS_URL, timeout=30)
                r.raise_for_status()
                p.write_text(r.text, encoding="utf-8")
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))["data"]
            else:
                with zipfile.ZipFile(config.ROOT / "cache" / "llm_cache.zip") as z:
                    data = json.loads(z.read("llm_cache/models_catalogue.json"))["data"]
            _prices = {m["id"]: (float(m["pricing"].get("prompt", 0) or 0), float(m["pricing"].get("completion", 0) or 0))
                       for m in data}
        except Exception:
            _prices = {}
        return _prices


def chat(model_id: str, messages, tools=None, tool_choice=None, temperature=None, max_tokens=None,
         extra=None, use_cache=True):
    

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or config.MAX_TOKENS,
        "usage": {"include": True},
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    if extra:
        payload.update(extra)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cp = _cache_path(key)
    nocache = os.environ.get("PHS_NO_CACHE") == "1"   
    if use_cache and not nocache and cp.exists():
        try:
            out = json.loads(cp.read_text(encoding="utf-8"))
            out["cached"] = True
            return out
        except (json.JSONDecodeError, OSError):
            pass  

    if use_cache and not nocache:
        archive = config.ROOT / "cache" / "llm_cache.zip"
        if archive.is_file():
            try:
                with zipfile.ZipFile(archive) as z:
                    out = json.loads(z.read(f"llm_cache/{key[:2]}/{key}.json"))
                out["cached"] = True
                return out
            except (KeyError, ValueError, OSError, zipfile.BadZipFile):
                pass

    if os.environ.get("PHS_MOCK_LLM") == "1":
        out = _mock(payload)
    else:
        out = _call(payload)
    if nocache:
        out["cached"] = False
        return out
    tmp = cp.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(tmp, cp)  
    except OSError:
        tmp.unlink(missing_ok=True)
    out["cached"] = False
    return out


_free_lock = threading.Lock()
_free_last = [0.0]


def _pace_free(model_id):
    
    if not str(model_id).endswith(":free"):
        return
    with _free_lock:
        wait = config.FREE_MIN_INTERVAL_S - (time.time() - _free_last[0])
        if wait > 0:
            time.sleep(wait)
        _free_last[0] = time.time()


def _call(payload):
    api_key = next((os.environ[k] for k in ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER_API")
                    if os.environ.get(k)), None)
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(config.MAX_RETRIES):
        _pace_free(payload["model"])
        t0 = time.perf_counter()
        try:
            r = requests.post(config.OPENROUTER_URL, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT)
            latency = time.perf_counter() - t0
            if r.status_code == 429 and "per-day" in r.text.lower().replace(" ", "-"):
                raise LLMError("FREE_DAILY_LIMIT: daily limit for free models reached - run again tomorrow. "
                               + r.text[:200])
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            if r.status_code == 404 and ("unavailable" in r.text.lower() or "no endpoints" in r.text.lower()):
                raise LLMError(f"MODEL_UNAVAILABLE: {r.text[:300]}")
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            if "error" in data:
                err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
                if err.get("code") in (400, 401, 402, 403) or "Policy Violation" in str(err.get("message", "")):
                    raise LLMError(f"Provider refused (not retried): {str(err)[:400]}")
                last = str(data["error"])[:300]
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            msg = data["choices"][0]["message"]
            u = data.get("usage", {}) or {}
            pt, ct = int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
            cost = u.get("cost")
            if cost is None:
                pp, cp_ = _prices_per_token().get(payload["model"], (0.0, 0.0))
                cost = pt * pp + ct * cp_
            return {"message": msg, "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cost": float(cost)},
                    "latency_s": latency, "provider": data.get("provider")}
        except (requests.Timeout, requests.ConnectionError) as e:
            last = repr(e)
            time.sleep(min(60, 2 ** attempt + random.random()))
    raise LLMError(f"Failed after {config.MAX_RETRIES} attempts: {last}")


def _mock(payload):
    
    msgs = payload["messages"]
    tools = [t["function"]["name"] for t in payload.get("tools", [])]
    used = {m.get("name") for m in msgs if m["role"] == "tool"}
    user_text = " ".join(m["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], str)).lower()
    forced = payload.get("tool_choice")
    pending = [t for t in tools if t != "submit_decision" and t not in used]
    if pending and not isinstance(forced, dict):
        name, args = pending[0], {}
    else:
        bad = any(w in user_text for w in ["kriminel", "hajdut", "qen", "mut", "tradhtar", "plehë", "pis"])
        name, args = "submit_decision", {"label": int(bad), "confidence": 0.7, "rationale": "mock decision"}
    msg = {"role": "assistant", "content": None, "tool_calls": [{
        "id": f"call_{len(msgs)}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)}}]}
    return {"message": msg, "usage": {"prompt_tokens": 300, "completion_tokens": 30, "cost": 0.0001},
            "latency_s": 0.01, "provider": "mock"}

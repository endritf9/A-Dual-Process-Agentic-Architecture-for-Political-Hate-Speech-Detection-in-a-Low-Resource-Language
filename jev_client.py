import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_INPUT_PRICE = 0.042 / 1_000_000
CACHE = Path(__file__).resolve().parent / "results" / "jev_cache"

PHS_DEFINITION_TRUE = (
    "Shows hostility or contempt toward a political target (a politician, a party or its members, the "
    "government or another state institution, or a group defined in political or national terms): insults, "
    "vulgar language, slurs, derogatory nicknames, mocking or sarcastic attacks, dehumanising words, "
    "accusations used as insults (thief, criminal, traitor), threats, wishes of harm, calls for hostility."
)
PHS_DEFINITION_FALSE = (
    "Neutral mention of politicians or institutions, news-like statement, question, support or praise, "
    "criticism of decisions or policies without insults, mockery or contempt, or no political target at all."
)


def phs_questions():
    
    return {
        "political_hate_speech": {
            "type": "noul",
            "instructions": "This is a social media comment in Albanian (standard or Gheg, often without "
                            "diacritics). Is it political hate speech?",
            "criteria": {"true": PHS_DEFINITION_TRUE, "false": PHS_DEFINITION_FALSE},
        },
        "target": {
            "type": "choice",
            "instructions": "Who or what is the main target of the comment?",
            "criteria": {
                "politician": "A named or clearly identified politician or office holder",
                "party": "A political party or its members and voters",
                "institution": "Government, parliament, police, courts or another state institution",
                "group": "An ethnic, national or religious group",
                "none": "No political target; private person, general topic or nothing specific",
            },
        },
        "hostility": {
            "type": "score",
            "instructions": "How hostile is the comment toward its target?",
            "criteria": [
                "Neutral or positive",
                "Critical but civil",
                "Rude or mocking",
                "Insulting or dehumanising",
                "Threatening or calling for harm",
            ],
        },
    }


def ask(state: str, questions: dict, use_cache=True, retries=5):
    
    if os.environ.get("PHS_MOCK_LLM") == "1":
        return _mock(state, questions)
    payload = {"state": state, "model": JEV_MODEL, "questions": questions}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cp = CACHE / h[:2] / f"{h}.json"
    nocache = os.environ.get("PHS_NO_CACHE") == "1"   
    if use_cache and not nocache and cp.exists():
        try:
            out = json.loads(cp.read_text(encoding="utf-8"))
            out["cached"] = True
            return out
        except json.JSONDecodeError:
            pass
    if use_cache and not nocache:
        archive = Path(__file__).resolve().parent / "cache" / "jev_cache.zip"
        if archive.is_file():
            try:
                with zipfile.ZipFile(archive) as z:
                    out = json.loads(z.read(f"jev_cache/{h[:2]}/{h}.json"))
                out["cached"] = True
                return out
            except (KeyError, ValueError, OSError, zipfile.BadZipFile):
                pass
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set.")
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(JEV_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            lat = time.perf_counter() - t0
            u = data.get("usage", {}) or {}
            out = {"answers": data.get("answers", {}), "model": data.get("model"), "usage": u,
                   "cost_usd": (u.get("input_tokens", 0) or 0) * JEV_INPUT_PRICE, "latency_s": lat}
            if nocache:
                out["cached"] = False
                return out
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cp)
            out["cached"] = False
            return out
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 529):
                last = f"HTTP {e.code}: {txt}"
                time.sleep(min(30, 2 ** attempt + random.random()))
                continue
            raise RuntimeError(f"Jev HTTP {e.code}: {txt}")
        except (urllib.error.URLError, TimeoutError) as e:
            last = repr(e)
            time.sleep(min(30, 2 ** attempt + random.random()))
    raise RuntimeError(f"Jev failed after {retries} attempts: {last}")


def _mock(state, questions):
    
    bad = any(w in state.lower() for w in ["kriminel", "hajdut", "qen", "mut", "tradhtar", "pis"])
    ans = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            ans[name] = {"type": "noul", "noul": 0.9 if bad else 0.1}
        elif q["type"] == "choice":
            opts = list(q["criteria"])
            pick = opts[0] if bad else opts[-1]
            ans[name] = {"type": "choice", "choice": pick, "confidence": 0.8,
                         "probabilities": {o: (0.9 if o == pick else 0.1 / max(1, len(opts) - 1)) for o in opts}}
        else:
            ans[name] = {"type": "score", "score": 3.0 if bad else 0.5, "confidence": 0.5}
    return {"answers": ans, "model": "mock", "usage": {"input_tokens": 300}, "cost_usd": 300 * JEV_INPUT_PRICE,
            "latency_s": 0.3, "cached": False}

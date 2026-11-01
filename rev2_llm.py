import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
import run_llm_experiments as R
from agent import CONFIGS, run_case
from common import ensure_dirs, get_logger
from rev2_data import load

log = get_logger("rev2_llm")
lock = threading.Lock()


def scores(ds, key):
    out = {}
    for clf in config.REV2_CLASSIFIER.split("+"):
        f = config.CLASSIFIER_PREDS / clf / f"foldpost_{ds}.csv"
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run rev2_import_mbert.py (after the GPU run) and rev2_jev.py first.")
        for r in pd.read_csv(f).to_dict("records"):
            e = {"p": float(r["prob_phs"])}
            if isinstance(r.get("target"), str) and r.get("target"):
                e["target"] = r["target"]
            if r.get("hostility") == r.get("hostility") and r.get("hostility") is not None:
                e["hostility"] = round(float(r["hostility"]), 2)
            out.setdefault(str(r[key]), {})[clf] = e
    return out


def cases(ds):
    d, key = load(ds)
    sc = scores(ds, key)
    for r in d.to_dict("records"):
        yield {"id": str(r[key]), "text": str(r["text"]), "label": int(r["label"]),
               "group": r.get("group", ""), "is_reply": int(r.get("is_reply", 0) or 0),
               "parent_text": r["parent_text"] if isinstance(r.get("parent_text"), str) else "",
               "related_texts": r["related_texts"] if isinstance(r.get("related_texts"), str) else "[]",
               "classifier_scores": sc.get(str(r[key]), {}), "classifier_name": config.REV2_CLASSIFIER}


def run(ds, cfg, short, model_id, budget):
    out = config.LLM_RUNS / f"rev2-{ds}__{cfg}__{short}__foldpost.jsonl"
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(str(json.loads(line)["id"]))
            except Exception:
                pass
    todo = [c for c in cases(ds) if c["id"] not in done]
    if not todo:
        log.info(f"[skip] {out.name} complete")
        return
    log.info(f"[start] {out.name}: {len(todo)} items ({len(done)} done)")
    n, errors, cost, t0 = 0, 0, 0.0, time.time()
    workers = 3 if model_id.endswith(":free") else config.N_WORKERS

    def work(c):
        if budget.exceeded:
            return None
        rec = run_case(c, cfg, model_id)
        budget.add(rec["cost"])
        d = rec["decision"] or {}
        return {"id": c["id"], "label": c["label"], "group": c["group"], "pred": d.get("label"),
                "confidence": d.get("confidence"), "rationale": d.get("rationale"), "config": cfg, "model": short,
                "model_id": model_id, "dataset": ds, **{k: rec[k] for k in (
                    "tool_calls", "llm_calls", "cached_calls", "prompt_tokens", "completion_tokens", "cost",
                    "latency_s", "wall_s", "error")}}

    with ThreadPoolExecutor(max_workers=workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, c) for c in todo]
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            if row["error"] or row["pred"] is None:
                errors += 1
                err = str(row["error"])
                if errors <= 3:
                    log.warning(f"  error on {row['id']}: {err[:200]}")
                if "FREE_DAILY_LIMIT" in err or "MODEL_UNAVAILABLE" in err or (
                        errors >= 15 and errors > 2 * max(n, 1) and "429" in err):
                    log.warning(f"{short}: daily limit / unavailable / rate-limited. Skipping for now; "
                                "finished results retained.")
                    for fu in futs:
                        fu.cancel()
                    return
                continue
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n += 1; cost += row["cost"]
            if n % (20 if workers == 1 else 100) == 0:
                log.info(f"  {out.name}: {n}/{len(todo)}, cost ${cost:.3f}, {(time.time() - t0) / 60:.1f} min, errors {errors}")
    log.info(f"[done] {out.name}: {n} written, {errors} errors, cost ${cost:.3f}; total spent ${budget.spent:.2f} of ${budget.cap:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["rev2", "rev2_free", "rev2_nemo_paid", "rev2_bias"], default="rev2")
    args = ap.parse_args()
    ensure_dirs(config.LLM_RUNS)
    budget = R.Budget()
    log.info(f"Spent so far ${budget.spent:.2f} of ${budget.cap:.2f} (paper rule: max $6 in total)")
    models = {**config.CHEAP_MODELS, **config.FREE_MODELS}
    plan = {"rev2": config.REV2_PLAN, "rev2_free": config.REV2_FREE_PLAN, "rev2_nemo_paid": config.REV2_NEMO_PAID_PLAN, "rev2_bias": config.REV2_BIAS_PLAN}[args.plan]
    if args.plan == "rev2_nemo_paid":
        
        
        models["nemotron-3-ultra-free"] = config.NEMOTRON_PAID_ID
    for ds, cfg, short in plan:
        if budget.exceeded:
            log.warning("Budget cap reached; stopping.")
            break
        assert cfg in CONFIGS, cfg
        run(ds, cfg, short, models[short], budget)


if __name__ == "__main__":
    main()

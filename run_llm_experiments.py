import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
from agent import CONFIGS, run_case
from common import ensure_dirs, get_logger, parse_split, split_masks

log = get_logger("05_llm_experiments")
_write_lock = threading.Lock()


class Budget:
    
    def __init__(self):
        self.spent = 0.0

        for sp in config.RESULTS.glob("llm_runs_archive*/SPENT.json"):
            self.spent += float(json.loads(sp.read_text()).get("spent_usd_all_calls", 0))

        for f in config.CLASSIFIER_PREDS.glob("jev*/*.csv") if config.CLASSIFIER_PREDS.exists() else []:
            d = pd.read_csv(f)
            if "cost" in d:
                self.spent += float((d.loc[~d["cached"].astype(bool), "cost"] if "cached" in d else d["cost"]).sum())
        if config.LLM_RUNS.exists():
            for f in config.LLM_RUNS.glob("*.jsonl"):
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            self.spent += float(json.loads(line).get("cost", 0) or 0)
                        except Exception:
                            pass
        self.lock = threading.Lock()

    def add(self, x):
        with self.lock:
            self.spent += x

    @property
    def cap(self):
        return min(config.BUDGET_USD, getattr(config, "PAPER_HARD_CAP_USD", 6.0))

    @property
    def exceeded(self):
        return self.spent >= self.cap


BUDGET = None


def load_classifier_probs(dataset, fold):
    
    name = config.AGENT_CLASSIFIER
    key = "id" if dataset == "main" else "cf_id"
    scores = {}
    for clf in name.split("+"):
        f = config.CLASSIFIER_PREDS / clf / (f"fold{fold}_test.csv" if dataset == "main" else f"fold{fold}_{dataset}.csv")
        if not f.exists():
            raise SystemExit(f"Classifier predictions not found: {f}. Run the System 1 steps first "
                             f"(mbert_existing.py and run_jev.py).")
        d = pd.read_csv(f)
        for r in d.to_dict("records"):
            e = {"p": float(r["prob_phs"])}
            if isinstance(r.get("target"), str) and r.get("target"):
                e["target"] = r["target"]
            if r.get("hostility") == r.get("hostility") and r.get("hostility") is not None:
                e["hostility"] = round(float(r["hostility"]), 2)
            scores.setdefault(r[key], {})[clf] = e
    return scores, name


def few_shot_examples(df, fold):
    trm, _ = split_masks(df, fold)
    tr = df[trm & df.text.str.len().between(20, 300)]
    k = config.FEW_SHOT_K // 2
    shots = pd.concat([tr[tr.label == 1].sample(k, random_state=config.FEW_SHOT_SEED),
                       tr[tr.label == 0].sample(k, random_state=config.FEW_SHOT_SEED)])
    return shots.sample(frac=1, random_state=config.FEW_SHOT_SEED)[["text", "label"]].to_dict("records")


def cases_for(dataset, fold, df, cf):
    probs, cname = load_classifier_probs(dataset, fold)
    if dataset == "main":
        sub = df[split_masks(df, fold)[1]]
        for r in sub.itertuples(index=False):
            yield {"id": r.id, "text": r.text, "label": int(r.label), "is_reply": int(r.is_reply),
                   "parent_text": r.parent_text if isinstance(r.parent_text, str) else "",
                   "related_texts": r.related_texts if isinstance(r.related_texts, str) else "[]",
                   "classifier_scores": probs.get(r.id, {}), "classifier_name": cname}
    else:
        for r in cf.itertuples(index=False):
            yield {"id": r.cf_id, "text": r.text, "label": int(r.label), "group": r.group, "is_reply": 0,
                   "parent_text": "", "related_texts": "[]", "classifier_scores": probs.get(r.cf_id, {}),
                   "classifier_name": cname}


def stratified_sample(cases, n):
    
    d = pd.DataFrame({"i": range(len(cases)), "strat": [f"{c['label']}_{c.get('group', '')}" for c in cases]})
    frac = min(1.0, n / len(d))
    idx = d.groupby("strat", group_keys=False).apply(
        lambda g: g.sample(max(1, round(len(g) * frac)), random_state=config.SAMPLE_SEED))["i"]
    return [cases[i] for i in sorted(idx)]


def run(dataset, config_name, short, model_id, fold, df, cf, limit=None, sample=None):
    ensure_dirs(config.LLM_RUNS)
    tag = config_name
    if "get_classifier_score" in CONFIGS[config_name]["tools"]:
        tag += f"-{config.AGENT_CLASSIFIER}"
    if sample:
        tag += f"@s{sample}"
    out = config.LLM_RUNS / f"{dataset}__{tag}__{short}__fold{fold}.jsonl"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    all_cases = list(cases_for(dataset, fold, df, cf))
    if sample:
        all_cases = stratified_sample(all_cases, sample)
    cases = [c for c in all_cases if c["id"] not in done]
    if limit:
        cases = cases[:limit]
    if not cases:
        log.info(f"[skip] {out.name} complete")
        return
    shots = few_shot_examples(df, fold) if CONFIGS[config_name]["few_shot"] else None
    log.info(f"[start] {out.name}: {len(cases)} comments to run ({len(done)} already done)")
    t0, n, cost, errors = time.time(), 0, 0.0, 0

    def work(case):
        if BUDGET.exceeded:
            return None
        rec = run_case(case, config_name, model_id, shots)
        BUDGET.add(rec["cost"])
        d = rec["decision"] or {}
        row = {"id": case["id"], "label": case["label"], "pred": d.get("label"), "confidence": d.get("confidence"),
               "rationale": d.get("rationale"), "config": config_name, "model": short, "model_id": model_id,
               "classifier": case["classifier_name"] if "get_classifier_score" in CONFIGS[config_name]["tools"] else None,
               "fold": fold, "dataset": dataset, **{k: rec[k] for k in (
                   "tool_calls", "llm_calls", "cached_calls", "prompt_tokens", "completion_tokens", "cost",
                   "latency_s", "wall_s", "error")}}
        if "group" in case:
            row["group"] = case["group"]
        return row

    workers = 1 if str(model_id).endswith(":free") else config.N_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, c) for c in cases]
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            if row["error"] or row["pred"] is None:
                errors += 1
                if errors <= 5:
                    log.warning(f"  error on id {row['id']}: {row['error']}")
                if row["error"] and "OPENROUTER_API_KEY" in row["error"]:
                    raise SystemExit(row["error"])
                if row["error"] and "FREE_DAILY_LIMIT" in row["error"]:
                    log.warning("Daily limit for free models reached. Stopping; "
                                "(finished comments are kept).")
                    for fu in futs:
                        fu.cancel()
                    raise SystemExit(0)
                if row["error"] and "MODEL_UNAVAILABLE" in row["error"]:
                    log.warning(f"Model {model_id} is no longer offered on OpenRouter. Skipping it; "
                                "going on with the next item of the plan.")
                    for fu in futs:
                        fu.cancel()
                    return
                if errors >= 15 and errors > 2 * max(n, 1) and "429" in str(row["error"]):
                    log.warning(f"{model_id} is rate-limited by its provider (many HTTP 429 errors). Skipping it "
                                "for now; finished results retained.")
                    for fu in futs:
                        fu.cancel()
                    return
                continue
            with _write_lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            n += 1
            cost += row["cost"]
            if n % (20 if workers == 1 else 100) == 0 or n == len(cases):
                log.info(f"  {out.name}: {n}/{len(cases)} done, cost so far ${cost:.3f}, "
                         f"{(time.time() - t0) / 60:.1f} min, errors {errors}")
    log.info(f"[done] {out.name}: {n} written, {errors} errors (errors are retried on next run), cost ${cost:.3f}; "
             f"total spent ${BUDGET.spent:.2f} of ${BUDGET.cap:.2f}")
    if BUDGET.exceeded:
        log.warning(f"BUDGET CAP REACHED (${BUDGET.spent:.2f}). Stopping. The research budget cap allows at most $" + f"{BUDGET.cap:.2f} in total.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["zero_shot", "few_shot", "agent_full"], choices=list(CONFIGS))
    ap.add_argument("--models", nargs="+", default=list(config.LLM_MODELS))
    ap.add_argument("--set", choices=list(config.MODEL_SETS), default=None,
                    help="named model set from config.MODEL_SETS (overrides --models)")
    ap.add_argument("--folds", nargs="+", type=parse_split, default=config.LLM_FOLDS,
                    help='fold ids 0-4 or "orig" (original 75/25 split)')
    ap.add_argument("--dataset", choices=["main", "cf", "ncf"], default="main")
    ap.add_argument("--plan", choices=["budget", "free", "revision", "revision_free"], default=None,
                    help="run config.BUDGET_PLAN in priority order on LLM_FOLDS[0], within BUDGET_USD")
    ap.add_argument("--limit", type=int, default=None, help="only the first N comments (for a quick try)")
    ap.add_argument("--sample", type=int, default=None,
                    help="fixed stratified subsample of about N comments (cheap proof of concept)")
    args = ap.parse_args()

    df = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    cf = pd.read_csv(config.COUNTERFACTUAL_CSV, encoding="utf-8")
    ncf = pd.read_csv(config.NATURAL_CF_CSV, encoding="utf-8") if config.NATURAL_CF_CSV.exists() else None
    pick = lambda ds: ncf if ds == "ncf" else cf
    all_models = {**config.LLM_MODELS, **config.STRONG_MODEL, **config.CHEAP_MODELS, **config.FREE_MODELS}
    if args.set:
        args.models = config.MODEL_SETS[args.set]
    global BUDGET
    BUDGET = Budget()
    log.info(f"Agent classifier tool uses: {config.AGENT_CLASSIFIER}. "
             f"Spent so far ${BUDGET.spent:.2f} of budget ${BUDGET.cap:.2f} (paper rule: max $6 in total)")
    if args.plan:
        plan = {"budget": config.BUDGET_PLAN, "free": config.FREE_PLAN, "revision": config.REVISION_PLAN,
                "revision_free": config.REVISION_FREE_PLAN}[args.plan]
        for dataset, cfg, short in plan:
            if BUDGET.exceeded:
                log.warning("Budget reached; remaining plan items skipped.")
                break
            run(dataset, cfg, short, all_models[short], config.LLM_FOLDS[0], df, pick(dataset), args.limit, args.sample)
        return
    for short in args.models:
        if short not in all_models:
            raise SystemExit(f"Unknown model '{short}'. Known: {list(all_models)}")
        for cfg in args.configs:
            for fold in (args.folds if args.dataset == "main" else args.folds[:1]):
                run(args.dataset, cfg, short, all_models[short], fold, df, pick(args.dataset), args.limit, args.sample)


if __name__ == "__main__":
    main()

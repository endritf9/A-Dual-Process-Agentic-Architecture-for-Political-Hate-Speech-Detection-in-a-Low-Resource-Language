import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
import jev_client as J
from common import binary_metrics, ensure_dirs, get_logger, split_masks

log = get_logger("13_run_jev")
N_WORKERS = 4
N_EXAMPLES_PER_CLASS = 4


def example_question(df, split):
    
    trm, _ = split_masks(df, split)
    tr = df[trm & df.text.str.len().between(25, 160)]
    rnd = random.Random(config.FEW_SHOT_SEED)
    ex = {y: rnd.sample(tr[tr.label == y]["text"].tolist(), N_EXAMPLES_PER_CLASS) for y in (0, 1)}
    return {"label": {
        "type": "choice",
        "instructions": "This is a social media comment in Albanian (standard or Gheg, often without diacritics). "
                        "Is it political hate speech?",
        "criteria": {
            "political_hate_speech": {"what": J.PHS_DEFINITION_TRUE, "not_for": J.PHS_DEFINITION_FALSE,
                                      "examples": ex[1]},
            "not_political_hate_speech": {"what": J.PHS_DEFINITION_FALSE, "not_for": J.PHS_DEFINITION_TRUE,
                                          "examples": ex[0]},
        },
    }}


def parse(variant, out):
    a = out["answers"]
    if variant == "jev":
        return {"prob_phs": float(a["political_hate_speech"]["noul"]), "target": a["target"]["choice"],
                "hostility": float(a["hostility"]["score"])}
    p = a["label"]["probabilities"]
    return {"prob_phs": float(p.get("political_hate_speech", 0.0)), "target": "", "hostility": float("nan")}


def run_set(variant, questions, items, out_path, key):
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)[key].astype(str))
    todo = [it for it in items if str(it[key]) not in done]
    if not todo:
        log.info(f"[skip] {out_path.name} ({variant}) complete")
        return
    log.info(f"[start] {variant} {out_path.name}: {len(todo)} items ({len(done)} already done)")
    lock = threading.Lock()
    rows, errors = [], 0

    def work(it):
        out = J.ask(it["text"], questions)
        r = {key: it[key], "label": it["label"], **parse(variant, out),
             "latency_ms": round(1000 * out["latency_s"], 1), "cost": out["cost_usd"],
             "cached": out.get("cached", False)}
        if "group" in it:
            r["group"] = it["group"]
        return r

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.warning(f"  error: {e}")
                continue
            r["pred"] = int(r["prob_phs"] >= 0.5)
            with lock:
                rows.append(r)
            if i % 100 == 0:
                log.info(f"  {variant} {out_path.name}: {i}/{len(todo)}")
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(out_path) if out_path.exists() else None
        allr = pd.concat([old, new]) if old is not None else new
        allr.to_csv(out_path, index=False)
    log.info(f"[done] {variant} {out_path.name}: {len(rows)} new rows, {errors} errors "
             f"(errors are retried on the next run), cost ${sum(r['cost'] for r in rows if not r['cached']):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["jev", "jev_ex"], choices=["jev", "jev_ex"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    split = config.LLM_FOLDS[0]
    df = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    cf = pd.read_csv(config.COUNTERFACTUAL_CSV, encoding="utf-8")
    te = df[split_masks(df, split)[1]]
    main_items = te[["id", "text", "label"]].to_dict("records")
    cf_items = cf[["cf_id", "text", "label", "group"]].to_dict("records")
    if args.limit:
        main_items, cf_items = main_items[:args.limit], cf_items[:args.limit]
    for variant in args.variants:
        q = J.phs_questions() if variant == "jev" else example_question(df, split)
        out_dir = config.CLASSIFIER_PREDS / variant
        ensure_dirs(out_dir)
        run_set(variant, q, main_items, out_dir / f"fold{split}_test.csv", "id")
        run_set(variant, q, cf_items, out_dir / f"fold{split}_cf.csv", "cf_id")
        if config.NATURAL_CF_CSV.exists():
            ncf = pd.read_csv(config.NATURAL_CF_CSV, encoding="utf-8")
            run_set(variant, q, ncf[["cf_id", "text", "label", "group"]].to_dict("records"),
                    out_dir / f"fold{split}_ncf.csv", "cf_id")
        t = pd.read_csv(out_dir / f"fold{split}_test.csv")
        m = binary_metrics(t["label"], t["pred"])
        log.info(f"{variant} on {len(t)} test comments: macro-F1 {m['macro_f1']:.2f}, PHS-F1 {m['phs_f1']:.2f}, "
                 f"mean latency {t['latency_ms'].mean():.0f} ms")


if __name__ == "__main__":
    main()

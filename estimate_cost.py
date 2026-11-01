import json

import pandas as pd

import config
from common import get_logger, split_masks
from llm_client import _prices_per_token

log = get_logger("estimate_cost")


GUESS = {"zero_shot": (600, 300), "few_shot": (900, 300), "agent_full": (4100, 900),
         "agent_no_classifier": (3500, 750), "agent_no_entities": (3500, 750), "agent_no_context": (3500, 750)}


def measured():
    rows = []
    for f in config.LLM_RUNS.glob("*.jsonl") if config.LLM_RUNS.exists() else []:
        dataset, system, model, _ = f.stem.split("__")
        cfg = system.split("@")[0].split("-")[0]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows.append({"file": f.stem, "dataset": dataset, "config": cfg, "model": model,
                             "cost": r.get("cost", 0) or 0})
    return pd.DataFrame(rows)


def main():
    m = measured()
    all_models = {**config.LLM_MODELS, **config.STRONG_MODEL, **config.CHEAP_MODELS}
    prices = _prices_per_token()
    n_main = int(split_masks(pd.read_csv(config.FOLDS_CSV), config.LLM_FOLDS[0])[1].sum())
    n_cf = len(pd.read_csv(config.COUNTERFACTUAL_CSV))
    out, total, spent = [], 0.0, float(m["cost"].sum()) if not m.empty else 0.0
    for sp in config.RESULTS.glob("llm_runs_archive*/SPENT.json"):
        spent += float(json.loads(sp.read_text()).get("spent_usd_all_calls", 0))
    for f in config.CLASSIFIER_PREDS.glob("jev*/*.csv") if config.CLASSIFIER_PREDS.exists() else []:
        d = pd.read_csv(f)
        if "cost" in d:
            spent += float((d.loc[~d["cached"].astype(bool), "cost"] if "cached" in d else d["cost"]).sum())
    for dataset, cfg, short in config.BUDGET_PLAN:
        n = n_main if dataset == "main" else n_cf
        sel = m[(m.config == cfg) & (m.model == short)] if not m.empty else m
        done = int(len(sel[sel.dataset == dataset])) if not m.empty else 0
        ref = m[(m.config == "agent_full") & (m.model == short)] if not m.empty else m
        if not m.empty and len(sel) >= 3:
            per, src = sel["cost"].mean(), f"measured ({len(sel)})"
        elif cfg.startswith("agent_no") and not m.empty and len(ref) >= 3:
            per, src = ref["cost"].mean(), "as agent_full"
        elif cfg == "few_shot" and not m.empty and len(m[(m.config == "zero_shot") & (m.model == short)]) >= 3:
            per, src = 1.6 * m[(m.config == "zero_shot") & (m.model == short)]["cost"].mean(), "1.6 x zero_shot"
        else:
            pp, cp = prices.get(all_models[short], (0, 0))
            pt, ct = GUESS[cfg]
            per, src = pt * pp + ct * cp, "estimate"
        remaining = per * max(0, n - done)
        total += remaining
        out.append({"dataset": dataset, "config": cfg, "model": short, "done": f"{done}/{n}",
                    "usd_per_comment": round(per, 5), "remaining_usd": round(remaining, 3), "basis": src,
                    "cumulative_usd": round(spent + total, 2)})
    t = pd.DataFrame(out)
    log.info("\n" + t.to_string(index=False))
    log.info(f"Spent so far: ${spent:.2f}. Projected total for the plan: ${spent + total:.2f} "
             f"(budget cap ${config.BUDGET_USD:.2f}). Items whose cumulative cost passes the cap will not run.")


if __name__ == "__main__":
    main()

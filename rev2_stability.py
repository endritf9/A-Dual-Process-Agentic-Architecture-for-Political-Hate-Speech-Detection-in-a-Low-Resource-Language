import json, os, time
os.environ["PHS_NO_CACHE"] = "1"
import pandas as pd
import config, jev_client as J
from agent import run_case
from run_jev import run_set
from rev2_llm import cases

OUT = config.RESULTS_DIR / "stability" if hasattr(config, "RESULTS_DIR") else config.LLM_RUNS.parent / "stability"
OUT.mkdir(parents=True, exist_ok=True)
day = time.strftime("%Y-%m-%d")
ids = set(pd.read_csv(config.REV2_DIR / "phs_post.csv").query("split == 'test'").sample(100, random_state=2026)["id"].astype(str))


items = [c for c in cases("ptest") if c["id"] in ids]
jf = OUT / f"{day}_jev.csv"
run_set("jev", J.phs_questions(), [{"id": c["id"], "text": c["text"], "label": c["label"]} for c in items], jf, "id")
orig = pd.read_csv(config.CLASSIFIER_PREDS / "jev" / "foldpost_ptest.csv").assign(id=lambda d: d["id"].astype(str)).set_index("id")
new = pd.read_csv(jf).assign(id=lambda d: d["id"].astype(str)).set_index("id")
jev_same = (orig.loc[new.index, "pred"] == new["pred"]).mean()
jev_dp = (orig.loc[new.index, "prob_phs"] - new["prob_phs"]).abs().mean()


af = OUT / f"{day}_agent.jsonl"
done = {json.loads(l)["id"] for l in open(af, encoding="utf-8")} if af.exists() else set()
with open(af, "a", encoding="utf-8") as f:
    for c in items:
        if c["id"] in done:
            continue
        rec = run_case(c, "agent_full", {**config.CHEAP_MODELS, **config.FREE_MODELS}["qwen3.7-flash"])
        d = rec["decision"] or {}
        f.write(json.dumps({"id": c["id"], "pred": d.get("label"), "cost": rec["cost"], "cached_calls": rec["cached_calls"]}) + "\n"); f.flush()
a = pd.DataFrame([json.loads(l) for l in open(af, encoding="utf-8")]).set_index("id")
o = pd.DataFrame([json.loads(l) for l in open(config.LLM_RUNS / "rev2-ptest__agent_full__qwen3.7-flash__foldpost.jsonl", encoding="utf-8")])
o = o.assign(id=o["id"].astype(str)).drop_duplicates("id", keep="last").set_index("id")
ag_same = (o.loc[a.index, "pred"] == a["pred"]).mean()
row = {"date": day, "n": len(a), "jev_same_decision": round(100 * jev_same, 1), "jev_mean_abs_prob_diff": round(jev_dp, 4),
       "agent_same_decision": round(100 * ag_same, 1), "agent_cached_calls": int(a["cached_calls"].sum()), "agent_cost": round(a["cost"].sum(), 4)}
s = OUT / "summary.csv"
pd.concat([pd.read_csv(s), pd.DataFrame([row])]) .to_csv(s, index=False) if s.exists() else pd.DataFrame([row]).to_csv(s, index=False)
print(row)

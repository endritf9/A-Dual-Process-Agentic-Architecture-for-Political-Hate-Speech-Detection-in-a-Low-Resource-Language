import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config
NICE = {"mbert_orig": "mBERT", "jev": "Jev", "jev_ex": "Jev with examples", "tfidf_lr": "TF-IDF LR", "agree": "mBERT + Jev"}
from common import binary_metrics, ensure_dirs, get_logger

log = get_logger("06_evaluate")


def pm(values, digits=2):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return ""
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"


def load_classifier(name, fold, kind="test"):
    f = config.CLASSIFIER_PREDS / name / f"fold{fold}_{kind}.csv"
    return pd.read_csv(f) if f.exists() else None


def classifier_names():
    if not config.CLASSIFIER_PREDS.exists():
        return []
    return sorted(p.name for p in config.CLASSIFIER_PREDS.iterdir() if p.is_dir())


def load_runs(dataset):
    rows = []
    if not config.LLM_RUNS.exists():
        return pd.DataFrame()
    for f in sorted(config.LLM_RUNS.glob(f"{dataset}__*.jsonl")):
        _, system, model, fold = f.stem.split("__")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                r["system"] = system
                r["fold"] = str(r.get("fold"))
                rows.append(r)
    return pd.DataFrame(rows)


def table_classifiers():
    rows = []
    for name in classifier_names():
        per = []
        for k in range(config.N_FOLDS):
            d = load_classifier(name, k)
            if d is not None:
                per.append(binary_metrics(d["label"], d["pred"]))
        if per:
            rows.append({"model": name, "split": f"{len(per)} folds",
                         **{m: pm([p[m] for p in per]) for m in ("macro_p", "macro_r", "macro_f1", "weighted_f1", "phs_f1")}})
        d = load_classifier(name, "orig")
        if d is not None:
            o = binary_metrics(d["label"], d["pred"])
            rows.append({"model": name, "split": "orig 75/25",
                         **{m: f"{o[m]:.2f}" for m in ("macro_p", "macro_r", "macro_f1", "weighted_f1", "phs_f1")}})
    return pd.DataFrame(rows)


def table_llm(runs):
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for (system, model), g in runs.groupby(["system", "model"]):
        per = [binary_metrics(gf["label"], gf["pred"]) for _, gf in g.groupby("fold")]
        tools = g["tool_calls"].map(len)
        use = lambda t: 100 * g["tool_calls"].map(lambda x: t in x).mean()
        rows.append({
            "system": system, "model": model, "folds": g["fold"].nunique(), "n": len(g),
            "macro_f1": pm([p["macro_f1"] for p in per]), "weighted_f1": pm([p["weighted_f1"] for p in per]),
            "phs_p": pm([p["phs_p"] for p in per]), "phs_r": pm([p["phs_r"] for p in per]),
            "phs_f1": pm([p["phs_f1"] for p in per]),
            "cost_per_1k_usd": round(1000 * g["cost"].mean(), 3),
            "latency_s_mean": round(g["latency_s"].mean(), 2), "latency_s_p95": round(g["latency_s"].quantile(0.95), 2),
            "tool_calls_mean": round(tools.mean(), 2),
            "use_classifier_%": round(use("get_classifier_score"), 1),
            "use_entities_%": round(use("lookup_political_entities"), 1),
            "use_context_%": round(use("get_thread_context"), 1),
        })
    return pd.DataFrame(rows).sort_values(["model", "system"])


def cf_scores(df):
    out = {}
    for grp in ("neutral", "critique", "place"):
        s = df[df["group"] == grp]
        out[f"FPR_{grp}_%"] = round(100 * (s["pred"] == 1).mean(), 1) if len(s) else None
    s = df[df["group"] == "hateful"]
    out["recall_hateful_%"] = round(100 * (s["pred"] == 1).mean(), 1) if len(s) else None
    out["n"] = len(df)
    return out


def table_counterfactual(cf_runs):
    rows = []
    for name in classifier_names():
        per = [load_classifier(name, k, "cf") for k in config.LLM_FOLDS]
        per = [p for p in per if p is not None]
        if per:
            sc = [cf_scores(p) for p in per]
            rows.append({"system": f"classifier:{name}", "model": "-",
                         **{k: pm([s[k] for s in sc], 1) for k in sc[0] if k != "n"}, "n": sc[0]["n"]})
    if not cf_runs.empty:
        for (system, model), g in cf_runs.groupby(["system", "model"]):
            rows.append({"system": system, "model": model, **cf_scores(g)})
    return pd.DataFrame(rows)


def cf_by_entity_type(cf_runs):
    
    cf = pd.read_csv(config.COUNTERFACTUAL_CSV)
    rows = []
    for name in classifier_names():
        d = load_classifier(name, config.LLM_FOLDS[0], "cf")
        if d is None:
            continue
        d = d.merge(cf[["cf_id", "entity_type"]], on="cf_id")
        for et, g in d[d.group == "neutral"].groupby("entity_type"):
            rows.append({"system": f"classifier:{name}", "model": "-", "entity_type": et,
                         "FPR_neutral_%": round(100 * g["pred"].mean(), 1)})
    if not cf_runs.empty:
        d = cf_runs.merge(cf[["cf_id", "entity_type"]], left_on="id", right_on="cf_id")
        for (system, model, et), g in d[d.group == "neutral"].groupby(["system", "model", "entity_type"]):
            rows.append({"system": system, "model": model, "entity_type": et,
                         "FPR_neutral_%": round(100 * (g["pred"] == 1).mean(), 1)})
    return pd.DataFrame(rows)


def classifier_ms(name):
    
    f = config.TABLES / "gpu_latency.json"
    if f.exists():
        lat = json.loads(f.read_text())
        if f"{name}_cpu_ms_per_comment" in lat:
            return lat[f"{name}_cpu_ms_per_comment"]
    d = load_classifier(name, config.LLM_FOLDS[0])
    if d is not None and "latency_ms" in d:
        return float(d["latency_ms"].mean())
    return 0.5 if name.startswith("tfidf") else float("nan")


def s1_cost_per_comment(name):
    d = load_classifier(name, config.LLM_FOLDS[0])
    return float(d["cost"].mean()) if d is not None and "cost" in d else 0.0


def system1_frame(split):
    
    base = None
    for name in ("mbert_orig", "jev", "jev_ex", "tfidf_lr"):
        d = load_classifier(name, split)
        if d is None:
            continue
        d = d[["id", "label", "prob_phs"]].rename(columns={"prob_phs": f"p_{name}"})
        base = d if base is None else base.merge(d.drop(columns="label"), on="id", how="inner")
    return base


def s1_order(m, variant):
    
    if variant == "agree":
        pm_, pj = m["p_mbert_orig"].values, m["p_jev"].values
        lm, lj = pm_ >= 0.5, pj >= 0.5
        disagree = lm != lj
        doubt = np.maximum(1 - 2 * np.abs(pm_ - 0.5), 1 - 2 * np.abs(pj - 0.5))
        key = disagree * 10 + doubt
        label = ((pm_ + pj) / 2 >= 0.5).astype(int)
        return np.argsort(-key, kind="stable"), label, disagree
    p = m[f"p_{variant}"].values
    return np.argsort(-(1 - 2 * np.abs(p - 0.5)), kind="stable"), (p >= 0.5).astype(int), None


def routing(runs, folds_df):
    

    if runs.empty:
        return pd.DataFrame(), {}
    split = config.LLM_FOLDS[0]
    s1 = system1_frame(split)
    if s1 is None:
        return pd.DataFrame(), {}
    rows, curves = [], {}
    agent_runs = runs[runs["system"].str.startswith("agent_full") & (runs["fold"] == str(split))]
    for (system, model), g in agent_runs.groupby(["system", "model"]):
        m = s1.merge(g[["id", "pred", "cost", "latency_s"]].rename(columns={"pred": "agent_pred"}), on="id", how="left")
        if "@s" in system:
            m = m[m["id"].isin(set(g["id"]))]
        missing = int(m["agent_pred"].isna().sum())
        m[["cost", "latency_s"]] = m[["cost", "latency_s"]].fillna(0)
        n = len(m)
        for variant in config.S1_VARIANTS:
            needed = ["p_mbert_orig", "p_jev"] if variant == "agree" else [f"p_{variant}"]
            if any(c not in m for c in needed):
                continue
            order, s1_label, disagree = s1_order(m, variant)
            mm = m.iloc[order].reset_index(drop=True)
            s1_label = s1_label[order]
            agent = mm["agent_pred"].fillna(pd.Series(s1_label)).astype(int).values
            parts = ["mbert_orig", "jev"] if variant == "agree" else [variant]
            s1_ms = max(classifier_ms(c) for c in parts)
            s1_usd = sum(s1_cost_per_comment(c) for c in parts)
            curve = []
            rates = list(range(0, 101, 5))
            if disagree is not None:
                rates.append(round(100 * disagree.mean(), 1))
            for rate in sorted(set(rates)):
                esc = np.zeros(n, bool)
                esc[: int(round(n * rate / 100))] = True
                pred = np.where(esc, agent, s1_label)
                met = binary_metrics(mm["label"], pred)
                curve.append({"system1": variant, "escalated_%": rate,
                              "policy": "disagreements only" if disagree is not None and rate == round(100 * disagree.mean(), 1)
                              else "most doubtful first",
                              "macro_f1": round(met["macro_f1"], 2), "phs_f1": round(met["phs_f1"], 2),
                              "cost_per_1k_usd": round(1000 * (s1_usd + mm["cost"].values[esc].sum() / n), 3),
                              "latency_ms_mean": round(s1_ms + 1000 * mm["latency_s"].values[esc].sum() / n, 1)})
            for r in curve:
                rows.append({"system2": system, "model": model, "n": n, "agent_missing": missing, **r})
            curves[f"{system}__{model}__{variant}"] = pd.DataFrame(curve)
    return pd.DataFrame(rows), curves


def table_system1():
    
    s1 = system1_frame(config.LLM_FOLDS[0])
    if s1 is None:
        return pd.DataFrame(), None
    rows = []
    for c in [c for c in s1.columns if c.startswith("p_")]:
        name = c[2:]
        pred = (s1[c] >= 0.5).astype(int)
        met = binary_metrics(s1["label"], pred)
        rows.append({"system1": name, "n": len(s1), "macro_f1": round(met["macro_f1"], 2),
                     "phs_f1": round(met["phs_f1"], 2), "ECE": round(ece(s1[c].values, s1["label"].values), 4),
                     "latency_ms": round(classifier_ms(name), 1),
                     "cost_per_1k_usd": round(1000 * s1_cost_per_comment(name), 4)})
    if "p_mbert_orig" in s1 and "p_jev" in s1:
        lm, lj = s1["p_mbert_orig"] >= 0.5, s1["p_jev"] >= 0.5
        agree = lm == lj
        acc_agree = ((lm == (s1["label"] == 1)) & agree).sum() / max(1, agree.sum())
        rows.append({"system1": "agree (mBERT and Jev)", "n": len(s1),
                     "macro_f1": f"agree on {100 * agree.mean():.1f}% of comments",
                     "phs_f1": f"accuracy where they agree {100 * acc_agree:.1f}%",
                     "ECE": "", "latency_ms": "", "cost_per_1k_usd": ""})
    return pd.DataFrame(rows), s1


def ece(p, y, bins=10):
    
    p, y = np.asarray(p, float), np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if sel.any():
            e += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return e


def plot_calibration(s1):
    if s1 is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dirs(config.FIGURES)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="tab:gray", lw=0.8, label="perfect calibration")
    edges = np.linspace(0, 1, 11)
    for c, mk in zip([c for c in s1.columns if c.startswith("p_")], "osD^"):
        p, y = s1[c].values, s1["label"].values
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
            if sel.sum() >= 5:
                xs.append(p[sel].mean())
                ys.append(y[sel].mean())
        ax.plot(xs, ys, marker=mk, ms=4, label=NICE.get(c[2:], c[2:]))
    ax.set_xlabel("Predicted probability of political hate speech")
    ax.set_ylabel("Observed share of political hate speech")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "F_calibration_system1.png", dpi=200)
    plt.close(fig)


def plot_curves(curves):
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dirs(config.FIGURES)
    groups = {}
    for key, c in curves.items():
        run_key, variant = key.rsplit("__", 1)
        groups.setdefault(run_key, {})[variant] = c
    for run_key, vs in groups.items():
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for variant, c in vs.items():
            u = c[c.policy == "most doubtful first"]
            ax.plot(u["escalated_%"], u["macro_f1"], marker="o", ms=3, label=f"System 1: {NICE.get(variant, variant)}")
            d = c[c.policy == "disagreements only"]
            if not d.empty:
                ax.scatter(d["escalated_%"], d["macro_f1"], marker="*", s=90, zorder=4,
                           label="mBERT+Jev, disagreements only")
        ax.set_xlabel("Comments escalated to System 2 (%)")
        ax.set_ylabel("Macro-F1 (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(config.FIGURES / f"F_routing_{run_key}.png", dpi=200)
        plt.close(fig)


def mcnemar(runs):
    from scipy.stats import binomtest
    rows = []
    if runs.empty:
        return pd.DataFrame()
    for (system, model), g in runs.groupby(["system", "model"]):
        for clf in [c for c in classifier_names() if c in ("mbert_orig", "jev", "jev_ex", "tfidf_nb")]:
            parts = []
            for k, gf in g.groupby("fold"):
                c = load_classifier(clf, k)
                if c is not None:
                    parts.append(c.merge(gf[["id", "pred"]].rename(columns={"pred": "sys_pred"}), on="id"))
            if not parts:
                continue
            m = pd.concat(parts)
            a = ((m.sys_pred == m.label) & (m.pred != m.label)).sum()
            b = ((m.sys_pred != m.label) & (m.pred == m.label)).sum()
            p = binomtest(int(a), int(a + b), 0.5).pvalue if a + b else 1.0
            rows.append({"system": system, "model": model, "vs_classifier": clf, "n": len(m),
                         "system_only_correct": int(a), "classifier_only_correct": int(b), "p_value": round(p, 4)})
    return pd.DataFrame(rows)


def main():
    ensure_dirs(config.TABLES, config.FIGURES)
    folds_df = pd.read_csv(config.FOLDS_CSV)
    runs, cf_runs = load_runs("main"), load_runs("cf")
    tables = {
        "T1_classifiers": table_classifiers(),
        "T2_llm_and_agent": table_llm(runs),
        "T3_counterfactual": table_counterfactual(cf_runs),
        "T3b_counterfactual_by_entity_type": cf_by_entity_type(cf_runs),
    }
    t4, curves = routing(runs, folds_df)
    tables["T4_routing"] = t4
    t6, s1 = table_system1()
    tables["T6_system1"] = t6
    plot_calibration(s1)
    tables["T5_mcnemar"] = mcnemar(runs)
    plot_curves(curves)
    import formal_analysis
    formal_analysis.run_all(tables, s1, runs, cf_runs)

    md = ["# Results summary", "",
          f"Agent classifier tool: `{config.AGENT_CLASSIFIER}`. Values are % unless marked; ± is std over folds.", ""]
    for name, t in tables.items():
        t.to_csv(config.TABLES / f"{name}.csv", index=False)
        md += [f"## {name}", "", t.to_markdown(index=False) if not t.empty else "_no data yet_", ""]
    Path(config.TABLES / "summary.txt").write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote results/tables/summary.txt and the CSV tables")
    for name in ("T1_classifiers", "T2_llm_and_agent", "T3_counterfactual"):
        if not tables[name].empty:
            log.info(f"{name}:\n" + tables[name].to_string(index=False))


if __name__ == "__main__":
    main()

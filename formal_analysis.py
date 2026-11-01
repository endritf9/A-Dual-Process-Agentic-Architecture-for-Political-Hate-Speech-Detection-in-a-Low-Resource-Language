import numpy as np
import pandas as pd

import config
NICE = {"mbert_orig": "mBERT", "jev": "Jev", "jev_ex": "Jev with examples", "tfidf_lr": "TF-IDF LR", "agree": "mBERT + Jev"}
from common import ensure_dirs

B_BOOT = 1000
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def macro_f1(y, p):
    
    y, p = np.asarray(y, int), np.asarray(p, int)
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f))


def bootstrap_ci(y, pred, b=B_BOOT, seed=0):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        vals.append(macro_f1(y[idx], pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def yule_q(c_a, c_b):
    
    n11 = int(np.sum(c_a & c_b)); n00 = int(np.sum(~c_a & ~c_b))
    n10 = int(np.sum(c_a & ~c_b)); n01 = int(np.sum(~c_a & c_b))
    den = n11 * n00 + n01 * n10
    q = (n11 * n00 - n01 * n10) / den if den else float("nan")
    return n11, n10, n01, n00, q


def table_complementarity(s1):
    if s1 is None or "p_mbert_orig" not in s1 or "p_jev" not in s1:
        return pd.DataFrame()
    y = s1["label"].values
    rows = []
    for other in [c for c in ("p_jev", "p_jev_ex", "p_tfidf_lr") if c in s1]:
        ca = (s1["p_mbert_orig"].values >= 0.5) == (y == 1)
        cb = (s1[other].values >= 0.5) == (y == 1)
        n11, n10, n01, n00, q = yule_q(ca, cb)
        n = len(y)
        ea, eb = 1 - ca.mean(), 1 - cb.mean()
        rows.append({"pair": f"mbert_orig vs {other[2:]}", "n": n, "N11_both_right": n11, "N10_only_mbert": n10,
                     "N01_only_other": n01, "N00_both_wrong": n00, "Yule_Q": round(q, 3),
                     "double_fault_%": round(100 * n00 / n, 2),
                     "double_fault_if_independent_%": round(100 * ea * eb, 2),
                     "disagreement_%": round(100 * (n10 + n01) / n, 2)})
    return pd.DataFrame(rows)


def table_calibration(s1):
    if s1 is None:
        return pd.DataFrame()
    from evaluate import ece
    rows = []
    for c in [c for c in s1.columns if c.startswith("p_")]:
        rows.append({"system1": c[2:], "ECE": round(ece(s1[c].values, s1["label"].values), 4),
                     "Brier": round(brier(s1[c].values, s1["label"].values), 4)})
    return pd.DataFrame(rows)


def table_bootstrap(s1, runs):
    rows = []
    if s1 is not None:
        for c in [c for c in s1.columns if c.startswith("p_")]:
            pred = (s1[c] >= 0.5).astype(int)
            lo, hi = bootstrap_ci(s1["label"], pred)
            rows.append({"system": f"System 1: {c[2:]}", "model": "-", "n": len(s1),
                         "macro_f1": round(macro_f1(s1["label"], pred), 2), "ci95_low": round(lo, 2),
                         "ci95_high": round(hi, 2)})
    if not runs.empty:
        r = runs[runs["fold"] == str(config.LLM_FOLDS[0])]
        for (system, model), g in r.groupby(["system", "model"]):
            lo, hi = bootstrap_ci(g["label"], g["pred"])
            rows.append({"system": system, "model": model, "n": len(g),
                         "macro_f1": round(macro_f1(g["label"], g["pred"]), 2), "ci95_low": round(lo, 2),
                         "ci95_high": round(hi, 2)})
    return pd.DataFrame(rows)


def table_cascade_formal(t4, s1, runs):
    
    from evaluate import s1_order, s1_cost_per_comment
    if t4 is None or t4.empty or s1 is None:
        return pd.DataFrame(), {}
    split = str(config.LLM_FOLDS[0])
    rows, fronts = [], {}
    for (system, model), g4 in t4.groupby(["system2", "model"]):
        g = runs[(runs["system"] == system) & (runs["model"] == model) & (runs["fold"] == split)]
        m = s1.merge(g[["id", "pred", "cost"]].rename(columns={"pred": "agent_pred"}), on="id", how="inner")
        if m.empty:
            continue
        n = len(m)
        y = m["label"].values
        for variant in sorted(g4["system1"].unique()):
            order, s1_label, disagree = s1_order(m, variant)
            yo, s1o = y[order], s1_label[order]
            ag = m["agent_pred"].values[order].astype(int)
            cost = m["cost"].values[order]
            s1_right, ag_right = s1o == yo, ag == yo
            acc_s1 = s1_right.mean()
            parts = ["mbert_orig", "jev"] if variant == "agree" else [variant]
            s1_usd = sum(s1_cost_per_comment(c) for c in parts)

            pts = [0.1, 0.2, 0.3] + ([disagree.mean()] if disagree is not None else [])
            for r in pts:
                k = int(round(n * r))
                esc = np.zeros(n, bool); esc[:k] = True
                pred = np.where(esc, ag, s1o)
                rows.append({"system2": system, "model": model, "system1": variant,
                             "point": "disagreements" if disagree is not None and r == pts[-1] else f"r={r:.0%}",
                             "escalated_%": round(100 * r, 1), "macro_f1": round(macro_f1(yo, pred), 2),
                             "acc_S1_%": round(100 * acc_s1, 2), "acc_cascade_%": round(100 * (pred == yo).mean(), 2),
                             "acc_S1_on_escalated_%": round(100 * s1_right[esc].mean(), 2) if k else None,
                             "acc_S2_on_escalated_%": round(100 * ag_right[esc].mean(), 2) if k else None,
                             "upper_bound_perfect_S2_%": round(100 * (acc_s1 + min(r, 1 - acc_s1)), 2),
                             "cost_per_1k_usd": round(1000 * (s1_usd + cost[esc].sum() / n), 4)})

            esc = (~s1_right) & ag_right
            pred = np.where(esc, ag, s1o)
            rows.append({"system2": system, "model": model, "system1": variant, "point": "oracle router",
                         "escalated_%": round(100 * esc.mean(), 1), "macro_f1": round(macro_f1(yo, pred), 2),
                         "acc_S1_%": round(100 * acc_s1, 2), "acc_cascade_%": round(100 * (pred == yo).mean(), 2),
                         "acc_S1_on_escalated_%": 0.0 if esc.any() else None,
                         "acc_S2_on_escalated_%": 100.0 if esc.any() else None,
                         "upper_bound_perfect_S2_%": None,
                         "cost_per_1k_usd": round(1000 * (s1_usd + cost[esc].sum() / n), 4)})

        pts = g4[["system1", "escalated_%", "policy", "macro_f1", "cost_per_1k_usd"]].copy()
        pts = pts.sort_values(["cost_per_1k_usd", "macro_f1"], ascending=[True, False])
        best, flags = -1.0, []
        for f in pts["macro_f1"]:
            flags.append(f > best)
            best = max(best, f)
        pts["pareto"] = flags
        fronts[f"{system}__{model}"] = pts
        for b in BUDGETS:
            ok = pts[pts["cost_per_1k_usd"] <= b]
            if ok.empty:
                continue
            top = ok.sort_values("macro_f1", ascending=False).iloc[0]
            rows.append({"system2": system, "model": model, "system1": top["system1"],
                         "point": f"best within ${b}/1k", "escalated_%": top["escalated_%"],
                         "macro_f1": top["macro_f1"], "cost_per_1k_usd": top["cost_per_1k_usd"]})
    return pd.DataFrame(rows), fronts


def table_throughput(t4, runs):
    from evaluate import classifier_ms
    t_m = classifier_ms("mbert_orig")
    if not t_m or t_m != t_m:
        return pd.DataFrame()
    theta_s1 = 1000.0 / t_m
    rows = [{"component": "System 1 (mBERT, one CPU core)", "escalated_%": None,
             "comments_per_s": round(theta_s1, 1), "comments_per_hour": int(3600 * theta_s1)}]
    if not runs.empty:
        a = runs[runs["system"].str.startswith("agent_full")]
        for model, g in a.groupby("model"):
            t2 = g["latency_s"].mean()
            theta_s2 = config.N_WORKERS / t2 if t2 else float("nan")
            for r in (0.1, 0.2, 0.3, 0.5, 1.0):
                th = min(theta_s1, theta_s2 / r)
                rows.append({"component": f"cascade, agent {model} ({config.N_WORKERS} parallel calls)",
                             "escalated_%": int(100 * r), "comments_per_s": round(th, 2),
                             "comments_per_hour": int(3600 * th)})
    return pd.DataFrame(rows)


def table_entity_sensitivity(cf_runs):
    

    cf = pd.read_csv(config.COUNTERFACTUAL_CSV)
    cf = cf[cf["group"].isin(["neutral", "critique", "hateful"])]
    from evaluate import classifier_names, load_classifier
    rows = []
    for name in classifier_names():
        d = load_classifier(name, config.LLM_FOLDS[0], "cf")
        if d is None:
            continue
        d = d.merge(cf[["cf_id", "template_id", "entity_type"]], on="cf_id")
        gr = d.groupby("template_id")
        rows.append({"system": f"classifier:{name}", "model": "-", "templates": gr.ngroups,
                     "flip_rate_%": round(100 * gr["pred"].nunique().gt(1).mean(), 1),
                     "mean_prob_range": round((gr["prob_phs"].max() - gr["prob_phs"].min()).mean(), 3)})
    if cf_runs is not None and not cf_runs.empty:
        d = cf_runs.merge(cf[["cf_id", "template_id"]], left_on="id", right_on="cf_id")
        for (system, model), g in d.groupby(["system", "model"]):
            gr = g.groupby("template_id")
            rows.append({"system": system, "model": model, "templates": gr.ngroups,
                         "flip_rate_%": round(100 * gr["pred"].nunique().gt(1).mean(), 1), "mean_prob_range": None})
    return pd.DataFrame(rows)


def table_cascade_counterfactual(cf_runs):
    
    from evaluate import load_classifier
    if cf_runs is None or cf_runs.empty:
        return pd.DataFrame()
    split = config.LLM_FOLDS[0]
    m, j = load_classifier("mbert_orig", split, "cf"), load_classifier("jev", split, "cf")
    if m is None or j is None:
        return pd.DataFrame()
    cf = pd.read_csv(config.COUNTERFACTUAL_CSV)[["cf_id", "group"]]
    d = cf.merge(m[["cf_id", "prob_phs"]].rename(columns={"prob_phs": "pm"}), on="cf_id") \
          .merge(j[["cf_id", "prob_phs"]].rename(columns={"prob_phs": "pj"}), on="cf_id")
    rows = []

    def rep(name, model, pred, esc):
        r = {"system": name, "model": model, "escalated_%": round(100 * float(np.mean(esc)), 1)}
        for g, lab in (("neutral", "FPR_neutral_%"), ("critique", "FPR_critique_%"), ("place", "FPR_place_%"),
                       ("hateful", "recall_hateful_%")):
            r[lab] = round(100 * float(np.mean(pred[(d["group"] == g).values] == 1)), 1)
        rows.append(r)

    lm, lj = (d["pm"] >= 0.5).values, (d["pj"] >= 0.5).values
    rep("System 1: mBERT", "-", lm.astype(int), np.zeros(len(d), bool))
    rep("System 1: mean of mBERT and Jev", "-", (((d["pm"] + d["pj"]) / 2) >= 0.5).astype(int).values, np.zeros(len(d), bool))
    for (system, model), g in cf_runs[cf_runs["system"].str.startswith("agent_full")].groupby(["system", "model"]):
        dd = d.merge(g[["id", "pred"]].rename(columns={"id": "cf_id", "pred": "ag"}), on="cf_id", how="left")
        ag = dd["ag"].fillna(-1).astype(int).values
        dis = lm != lj
        s1 = (((dd["pm"] + dd["pj"]) / 2) >= 0.5).astype(int).values
        rep("cascade: mBERT+Jev, disagreements escalated", model, np.where(dis & (ag >= 0), ag, s1), dis)
        doubt = 1 - 2 * np.abs(dd["pm"].values - 0.5)
        order = np.argsort(-doubt, kind="stable")
        for r in (0.1, 0.2, 0.3):
            esc = np.zeros(len(dd), bool); esc[order[: int(round(r * len(dd)))]] = True
            rep(f"cascade: mBERT router, r={int(100 * r)}%", model, np.where(esc & (ag >= 0), ag, lm.astype(int)), esc)
    return pd.DataFrame(rows)


def plot_pareto(fronts, t8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dirs(config.FIGURES)
    for key, pts in fronts.items():
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for v, g in pts.groupby("system1"):
            ax.scatter(g["cost_per_1k_usd"], g["macro_f1"], s=12, alpha=0.6, label=f"System 1: {NICE.get(v, v)}")
        fr = pts[pts["pareto"]].sort_values("cost_per_1k_usd")
        ax.step(fr["cost_per_1k_usd"], fr["macro_f1"], where="post", color="black", lw=1, label="Pareto front")
        system, model = key.split("__")
        orc = t8[(t8["system2"] == system) & (t8["model"] == model) & (t8["point"] == "oracle router")]
        if not orc.empty:
            ax.scatter(orc["cost_per_1k_usd"], orc["macro_f1"], marker="*", s=70, color="tab:red", label="oracle router")
        ax.set_xlabel("Cost per 1000 comments (USD)")
        ax.set_ylabel("Macro-F1 (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(config.FIGURES / f"F_pareto_{key}.png", dpi=200)
        plt.close(fig)


def run_all(tables, s1, runs, cf_runs):
    tables["T7_complementarity"] = table_complementarity(s1)
    t8, fronts = table_cascade_formal(tables.get("T4_routing"), s1, runs)
    tables["T8_cascade_formal"] = t8
    tables["T9_bootstrap"] = table_bootstrap(s1, runs)
    tables["T10_calibration"] = table_calibration(s1)
    tables["T11_throughput"] = table_throughput(tables.get("T4_routing"), runs)
    tables["T3c_entity_sensitivity"] = table_entity_sensitivity(cf_runs)
    tables["T12_cascade_counterfactual"] = table_cascade_counterfactual(cf_runs)
    if not t8.empty:
        plot_pareto(fronts, t8)

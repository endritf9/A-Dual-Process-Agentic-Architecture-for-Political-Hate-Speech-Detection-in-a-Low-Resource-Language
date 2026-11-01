import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

import config
import rev2_data
from common import binary_metrics

B = 2000
DELTA = 2.0
BUDGET = 0.20
RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
LLMS = ["qwen3.7-flash", "qwen3.8-flash", "gemma-4-31b", "nemotron-3-ultra-free"]
NO_JEV_S2 = [("qwen3.7-flash", "agent_no_classifier"), ("qwen3.7-flash", "single_no_classifier"), ("qwen3.7-flash", "zero_shot")]
rng_seed = 20261030


def clf(name, ds):
    f = config.CLASSIFIER_PREDS / name / f"foldpost_{ds}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    key = "id" if "id" in d.columns else "cf_id"
    d = d.rename(columns={key: "key"})
    d["key"] = d["key"].astype(str)
    return d.set_index("key")


def llm(ds, cfg, model):
    f = config.LLM_RUNS / f"rev2-{ds}__{cfg}__{model}__foldpost.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d = d[d["pred"].isin([0, 1])].drop_duplicates("id", keep="last")
    if "nemotron" in model:   
        d["cost"] = d["prompt_tokens"].fillna(0) * 0.6e-6 + d["completion_tokens"].fillna(0) * 2.4e-6
    d["key"] = d["id"].astype(str)
    return d.set_index("key")


def threads():
    
    from prepare_data import load_raw
    try:
        raw = load_raw()
    except FileNotFoundError:
        roots = pd.read_csv(config.DATA_DIR / "thread_roots.csv")
        return pd.Series(roots["root"].astype(str).values, index=roots["id"].astype(str))
    t = np.where(raw["Post"].notna(), raw["Post"].astype("Int64").astype(str),
                 raw["Reply"].astype(str).str.split("-").str[0])
    return pd.Series(t, index=raw["NrRen"].astype(str))


def mf1(y, p):
    
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100 * (f[0] + f[1]) / 2


_POS = {}


def cluster_index(groups, rng):
    
    groups = np.asarray(groups).astype(str)
    key = (len(groups), hash(groups.tobytes()))
    if key not in _POS:
        u = np.unique(groups)
        _POS[key] = (u, [np.flatnonzero(groups == g) for g in u])
    u, pos = _POS[key]
    pick = rng.integers(0, len(u), len(u))
    return np.concatenate([pos[i] for i in pick])


def boot_ci(y, preds, groups, fn=mf1, b=B):
    
    rng = np.random.default_rng(rng_seed)
    y = np.asarray(y); preds = [np.asarray(p) for p in preds]
    pts = [fn(y, p) for p in preds]
    sam = np.empty((b, len(preds)))
    for i in range(b):
        ix = cluster_index(groups, rng)
        sam[i] = [fn(y[ix], p[ix]) for p in preds]
    ci = [tuple(np.percentile(sam[:, j], [2.5, 97.5])) for j in range(len(preds))]
    dif = [tuple(np.percentile(sam[:, j] - sam[:, 0], [2.5, 97.5])) for j in range(len(preds))]
    return pts, ci, dif


def mcnemar(y, a, b):
    ca, cb = (np.asarray(a) == y), (np.asarray(b) == y)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(n01, n10), n, 0.5))


def ece(y, p, bins=10, mass=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if mass:
        edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -1e-9, 1 + 1e-9
    else:
        edges = np.linspace(0, 1, bins + 1); edges[0] = -1e-9
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


class S1:
    

    def __init__(self, ds, mname, jname, tname="tfidf_lr"):
        self.m, self.j, self.t = clf(mname, ds), clf(jname, ds), clf(tname, ds)
        self.keys = self.m.index

    def frame(self):
        f = pd.DataFrame({"y": self.m["label"].astype(int), "pm": self.m["prob_phs"], "yp_m": self.m["pred"].astype(int)})
        if self.j is not None:
            f["pj"] = self.j.reindex(f.index)["prob_phs"]
            f["yp_j"] = self.j.reindex(f.index)["pred"].astype(int)
            f["jcost"] = self.j.reindex(f.index)["cost"] if "cost" in self.j.columns else 0.0
            f["pmean"] = (f["pm"] + f["pj"]) / 2
            f["yp_mean"] = (f["pmean"] >= 0.5).astype(int)
        if self.t is not None:
            f["yp_t"] = self.t.reindex(f.index)["pred"].astype(int)
        return f


def route(f, router, rate):
    
    if router == "mbert":
        d = 1 - 2 * (f["pm"] - 0.5).abs(); base, jev = f["yp_m"], False
    elif router == "mean":
        d = 1 - 2 * (f["pmean"] - 0.5).abs(); base, jev = f["yp_mean"], True
    elif router == "disagree":
        esc = f["yp_m"] != f["yp_j"]
        return esc.values, f["yp_mean"].values, True
    elif router == "disagree_tfidf":
        esc = f["yp_m"] != f["yp_t"]
        return esc.values, f["yp_m"].values, False
    else:
        raise ValueError(router)
    k = int(round(rate * len(f)))
    esc = np.zeros(len(f), bool)
    if k:
        esc[np.argsort(-d.values, kind="stable")[:k]] = True
    return esc, base.values, jev


def cascade(f, s2, router, rate=0.0):
    esc, base, jev = route(f, router, rate)
    pred = base.copy()
    missing = 0
    cost = (f["jcost"].sum() if jev and "jcost" in f else 0.0)
    if s2 is not None and esc.any():
        s = s2.reindex(f.index)
        has = s["pred"].notna().values
        use = esc & has
        missing = int((esc & ~has).sum())
        pred[use] = s["pred"].values[use].astype(int)
        cost += s["cost"].values[use].sum()
    return pred, esc, 1000 * cost / len(f), missing


ROUTERS = [("disagree", 0.0)] + [(r, x) for r in ("mbert", "mean") for x in RATES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbert", default="mbert_post")
    ap.add_argument("--mbert_hn", default="mbert_post_hn")
    ap.add_argument("--jev", default="jev")
    ap.add_argument("--out", default="rev2")
    a = ap.parse_args()
    out = config.TABLES / a.out
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Revision 2 results (post-disjoint split)", ""]
    th = threads()

    sv, st = S1("pval", a.mbert, a.jev).frame(), S1("ptest", a.mbert, a.jev).frame()
    gt = th.reindex(st.index).fillna("x").values
    post = rev2_data.phs_post().assign(key=lambda d: d["id"].astype(str)).set_index("key")
    big = post["post_id"].value_counts().reindex(post[post.split == "test"].post_id.unique()).idxmax()


    rows = []
    systems = {"mBERT": st["yp_m"].values, "Jev": st["yp_j"].values, "Mean S1": st["yp_mean"].values}
    for n in ("tfidf_lr", "tfidf_svm", "tfidf_nb", "tfidf_rf", a.mbert_hn):
        c = clf(n, "ptest")
        if c is not None:
            systems[n] = c.reindex(st.index)["pred"].astype(int).values
    for m in LLMS:
        for cfg in ("zero_shot", "agent_full", "single_all", "agent_no_classifier", "single_no_classifier"):
            r = llm("ptest", cfg, m)
            if r is not None and len(r) >= 0.95 * len(st):
                p = r.reindex(st.index)["pred"]
                fill = st["yp_m"].where(p.isna(), p).astype(int)   
                systems[f"{m} {cfg}"] = fill.values
                lv = r.loc[r["cached_calls"].fillna(0) == 0, "wall_s"]      
                rows.append({"system": f"{m} {cfg}", "missing": int(p.isna().sum()), "n_uncached": len(lv),
                             "lat_median": lv.median(), "lat_q1": lv.quantile(.25),
                             "lat_q3": lv.quantile(.75), "lat_p95": lv.quantile(.95),
                             "cost_per_1000": 1000 * r["cost"].mean()})
    lat = pd.DataFrame(rows)
    lat.to_csv(out / "llm_latency_ptest.csv", index=False)
    names = list(systems)
    pts, ci, dif = boot_ci(st["y"].values, [systems[n] for n in names], gt)
    tab = []
    for i, n in enumerate(names):
        m = binary_metrics(st["y"], systems[n])
        tab.append({"system": n, "macro_f1": pts[i], "ci_lo": ci[i][0], "ci_hi": ci[i][1], "phs_f1": m["phs_f1"],
                    "diff_vs_mbert_lo": dif[i][0], "diff_vs_mbert_hi": dif[i][1],
                    "mcnemar_p_vs_mbert": mcnemar(st["y"].values, systems[n], systems["mBERT"])})
    single = pd.DataFrame(tab)
    single.to_csv(out / "single_systems_ptest.csv", index=False)
    md += ["## Single systems on the test posts (macro-F1, thread-cluster bootstrap CI)", "", single.round(2).to_markdown(index=False), ""]


    pol = []
    for m in LLMS:
        v2, t2 = llm("pval", "agent_full", m), llm("ptest", "agent_full", m)
        if v2 is None or t2 is None:
            continue
        cand = []
        for router, rate in ROUTERS:
            p, esc, c, _ = cascade(sv, v2, router, rate)
            cand.append({"router": router, "rate": rate, "val_f1": mf1(sv["y"], p), "val_cost": c, "val_esc": esc.mean()})
        cand = pd.DataFrame(cand)
        cand.assign(llm=m).to_csv(out / f"validation_grid_{m}.csv", index=False)
        ok = cand[cand.val_cost <= BUDGET].sort_values(["val_f1", "val_cost"], ascending=[False, True])
        p2 = ok.iloc[0]
        for name, router, rate in (("P1", "disagree", 0.0), ("P2", p2.router, p2.rate)):
            p, esc, c, miss = cascade(st, t2, router, rate)
            vrow = cand[(cand.router == router) & (cand.rate == rate)].iloc[0]
            pol.append({"llm": m, "policy": name, "router": router, "rate": rate, "val_f1": vrow.val_f1,
                        "test_esc": esc.mean(), "test_cost": c, "missing": miss, "pred": p, "esc": esc})
    res = []
    for r in pol:
        pts, ci, dif = boot_ci(st["y"].values, [st["yp_m"].values, r["pred"]], gt)
        res.append({k: r[k] for k in ("llm", "policy", "router", "rate", "val_f1", "test_esc", "test_cost", "missing")} |
                   {"test_f1": pts[1], "ci_lo": ci[1][0], "ci_hi": ci[1][1], "mbert_f1": pts[0],
                    "diff_lo": dif[1][0], "diff_hi": dif[1][1], "non_inferior": dif[1][0] > -DELTA,
                    "mcnemar_p": mcnemar(st["y"].values, r["pred"], st["yp_m"].values)})
    pol_t = pd.DataFrame(res)
    pol_t.to_csv(out / "policies_test.csv", index=False)
    md += [f"## Frozen policies on the test posts (non-inferiority margin {DELTA})", "", pol_t.round(3).to_markdown(index=False), ""]

    
    tr_txt = set(post[post.split == "train"].text.fillna("").str.strip().str.lower())
    keep_big = (post.reindex(st.index).post_id != big).values
    keep_dup = ~post.reindex(st.index).text.fillna("").str.strip().str.lower().isin(tr_txt).values
    sens = []
    for lab, keep in (("without largest post", keep_big), ("without duplicates", keep_dup)):
        y = st["y"].values[keep]
        sens.append({"subset": lab, "n": int(keep.sum()), "mBERT": mf1(y, st["yp_m"].values[keep]),
                     **{f"{r['llm']} {r['policy']}": mf1(y, r["pred"][keep]) for r in pol}})
    pd.DataFrame(sens).to_csv(out / "sensitivity_ptest.csv", index=False)
    md += ["## Sensitivity (macro-F1)", "", pd.DataFrame(sens).round(2).to_markdown(index=False), ""]

    pp = []
    for pid in sorted(post.reindex(st.index).post_id.unique()):
        k = (post.reindex(st.index).post_id == pid).values
        y = st["y"].values[k]
        row = {"post": pid, "n": int(k.sum()), "phs": int(y.sum()), "mBERT acc": 100 * (st["yp_m"].values[k] == y).mean(),
               "Jev acc": 100 * (st["yp_j"].values[k] == y).mean()}
        row.update({f"{r['llm']} {r['policy']} acc": 100 * (r["pred"][k] == y).mean() for r in pol})
        pp.append(row)
    pd.DataFrame(pp).to_csv(out / "per_post_ptest.csv", index=False)
    md += ["## Accuracy per test post (macro-F1 is not defined for posts without political HS)", "",
           pd.DataFrame(pp).round(1).to_markdown(index=False), ""]


    av = []
    for m in LLMS:
        ag, si = llm("ptest", "agent_full", m), llm("ptest", "single_all", m)
        if ag is None or si is None:
            continue
        both = st.index.intersection(ag.index).intersection(si.index)
        esc = pd.Series(route(st, "disagree", 0)[0], index=st.index)
        for lab, idx in (("all", both), ("escalated (P1)", both[esc.reindex(both).values])):
            y = st.loc[idx, "y"].values
            g = th.reindex(idx).fillna("x").values
            pts, ci, dif = boot_ci(y, [si.loc[idx, "pred"].astype(int).values, ag.loc[idx, "pred"].astype(int).values], g)
            av.append({"llm": m, "subset": lab, "n": len(idx), "single_all": pts[0], "agent_full": pts[1],
                       "diff_agent_minus_single_lo": dif[1][0], "diff_agent_minus_single_hi": dif[1][1],
                       "cost_single": 1000 * si.loc[idx, "cost"].mean(), "cost_agent": 1000 * ag.loc[idx, "cost"].mean()})
    pd.DataFrame(av).to_csv(out / "agent_vs_single.csv", index=False)
    md += ["## Agent vs single prompt with the same evidence", "", pd.DataFrame(av).round(2).to_markdown(index=False), ""]


    md += bias_tables(a, out, th)


    cal = []
    for n, p in (("mBERT", st["pm"]), ("Jev", st["pj"])):
        rng = np.random.default_rng(rng_seed)
        y, p = st["y"].values, p.values
        boots = []
        for _ in range(B):
            ix = cluster_index(gt, rng)
            boots.append((ece(y[ix], p[ix]), ece(y[ix], p[ix], mass=True), np.mean((p[ix] - y[ix]) ** 2)))
        bo = np.array(boots)
        cal.append({"model": n, "ece_width": ece(y, p), "ece_width_ci": "%.3f-%.3f" % tuple(np.percentile(bo[:, 0], [2.5, 97.5])),
                    "ece_mass": ece(y, p, mass=True), "ece_mass_ci": "%.3f-%.3f" % tuple(np.percentile(bo[:, 1], [2.5, 97.5])),
                    "brier": np.mean((p - y) ** 2), "brier_ci": "%.3f-%.3f" % tuple(np.percentile(bo[:, 2], [2.5, 97.5]))})
    rng = np.random.default_rng(rng_seed); y = st["y"].values; pm_, pj_ = st["pm"].values, st["pj"].values
    dd = []
    for _ in range(B):
        ix = cluster_index(gt, rng)
        dd.append((ece(y[ix], pm_[ix]) - ece(y[ix], pj_[ix]), ece(y[ix], pm_[ix], mass=True) - ece(y[ix], pj_[ix], mass=True)))
    dd = np.array(dd)
    cal.append({"model": "difference mBERT - Jev", "ece_width": ece(y, pm_) - ece(y, pj_),
                "ece_width_ci": "%.3f-%.3f" % tuple(np.percentile(dd[:, 0], [2.5, 97.5])),
                "ece_mass": ece(y, pm_, mass=True) - ece(y, pj_, mass=True),
                "ece_mass_ci": "%.3f-%.3f" % tuple(np.percentile(dd[:, 1], [2.5, 97.5]))})
    
    lv = np.log(np.clip(sv["pm"], 1e-6, 1 - 1e-6) / (1 - np.clip(sv["pm"], 1e-6, 1 - 1e-6)))
    lt = np.log(np.clip(st["pm"], 1e-6, 1 - 1e-6) / (1 - np.clip(st["pm"], 1e-6, 1 - 1e-6)))
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 400))
    nll = [-np.mean(sv["y"] * np.log(1 / (1 + np.exp(-lv / T))) + (1 - sv["y"]) * np.log(1 - 1 / (1 + np.exp(-lv / T)) + 1e-12))
           for T in grid]
    T = grid[int(np.argmin(nll))]
    pts_ = 1 / (1 + np.exp(-lt / T))
    mean_ts = (pts_ + st["pj"]) / 2
    cal.append({"model": f"mBERT temperature-scaled (T={T:.2f}, fitted on validation)", "ece_width": ece(st["y"], pts_),
                "ece_mass": ece(st["y"], pts_, mass=True), "brier": np.mean((pts_ - st["y"]) ** 2),
                "router_changes_mean_doubt": "n/a"})
    k = int(round(0.2 * len(st)))
    a1 = set(np.argsort(-(1 - 2 * (st["pmean"] - 0.5).abs()).values, kind="stable")[:k])
    a2 = set(np.argsort(-(1 - 2 * (mean_ts - 0.5).abs()).values, kind="stable")[:k])
    cal[-1]["router_changes_mean_doubt"] = f"{len(a1 ^ a2) // 2} of {k} escalated comments change at 20%"
    pd.DataFrame(cal).to_csv(out / "calibration_ptest.csv", index=False)
    md += ["## Calibration on the test posts", "", pd.DataFrame(cal).to_markdown(index=False), ""]

    md += cv_summary(a, out)


    f = config.TABLES / "rev2_mbert_seeds.csv"
    if f.exists():
        s = pd.read_csv(f)
        md += ["## mBERT over seeds", "", s.groupby("variant")[["pval_macro_f1", "ptest_macro_f1", "hn_fpr"]]
               .agg(["mean", "std"]).round(2).to_markdown(), ""]
    md += ["## LLM latency per comment on the test posts (s)", "", lat.round(2).to_markdown(index=False), ""]
    (out / "summary.txt").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def bias_tables(a, out, th):
    md = []
    
    hn = S1("hn", a.mbert, a.jev).frame()
    rows = [("mBERT", hn["yp_m"]), ("Jev", hn["yp_j"]), ("Mean S1", hn["yp_mean"])]
    c = clf(a.mbert_hn, "hn")
    if c is not None:
        rows.append(("mBERT trained with hard negatives", c.reindex(hn.index)["pred"]))
    for m in LLMS:
        for cfg in ("zero_shot", "agent_full", "single_all", "agent_no_classifier", "single_no_classifier"):
            r = llm("hn", cfg, m)
            if r is not None and len(r) >= 0.95 * len(hn):
                rows.append((f"{m} {cfg}", r.reindex(hn.index)["pred"]))
                if cfg in ("agent_full", "single_all"):
                    p, esc, _, _ = cascade(hn, r, "disagree")
                    rows.append((f"cascade P1 + {m} {cfg}", pd.Series(p, index=hn.index)))
    keep = keep_hn(hn.index)
    t = []
    for n, p in rows:
        p = pd.Series(p, index=hn.index)[keep].dropna().astype(float)
        g = th.reindex(p.index).fillna("x").values
        p = p.values
        rng = np.random.default_rng(rng_seed)
        bo = [p[cluster_index(g, rng)].mean() for _ in range(B)]
        t.append({"system": n, "n": len(p), "fpr": 100 * p.mean(), "ci_lo": 100 * np.percentile(bo, 2.5),
                  "ci_hi": 100 * np.percentile(bo, 97.5)})
    t = pd.DataFrame(t)
    t.to_csv(out / "hard_negatives_fpr.csv", index=False)
    md += [f"## False positive rate on hard negatives (n = {int(keep.sum())} confirmed non-HS)", "", t.round(1).to_markdown(index=False), ""]

    
    cf2 = rev2_data.load("ncf2")[0].assign(key=lambda d: d["cf_id"].astype(str)).set_index("key")
    acc = accepted_pairs()
    s1 = S1("ncf2", a.mbert, a.jev).frame()
    sysm = {"mBERT": s1["yp_m"], "Jev": s1["yp_j"], "Mean S1": s1["yp_mean"], "always PHS": s1["yp_m"] * 0 + 1}
    c = clf(a.mbert_hn, "ncf2")
    if c is not None:
        sysm["mBERT trained with hard negatives"] = c.reindex(s1.index)["pred"]
    for m in LLMS:
        for cfg in ("zero_shot", "agent_full", "single_all", "agent_no_classifier", "single_no_classifier"):
            r = llm("ncf2", cfg, m)
            if r is not None and len(r) >= 0.95 * len(s1):
                sysm[f"{m} {cfg}"] = r.reindex(s1.index)["pred"]
                if cfg in ("agent_full", "single_all", "agent_no_classifier"):
                    p, _, _, _ = cascade(s1, r, "disagree")
                    sysm[f"cascade P1 + {m} {cfg}"] = pd.Series(p, index=s1.index)
    t = []
    for n, p in sysm.items():
        t += flip_table(n, p, cf2, acc)
    t = pd.DataFrame(t)
    t.to_csv(out / "natural_cf2.csv", index=False)
    md += [f"## Natural counterfactuals v2 ({'validated pairs only' if acc is not None else 'NOT YET VALIDATED: all pairs'})",
           "", t.round(1).to_markdown(index=False), ""]

    
    t = []
    for ds in ("cf", "gen"):
        s = S1(ds, a.mbert, a.jev).frame()
        d = rev2_data.load(ds)[0].assign(key=lambda x: x["cf_id"].astype(str)).set_index("key").reindex(s.index)
        hate = d["group"].str.contains("hate").values
        for n, p in (("mBERT", s["yp_m"]), ("Jev", s["yp_j"]), ("mBERT trained with hard negatives",
                                                                  None if clf(a.mbert_hn, ds) is None else clf(a.mbert_hn, ds).reindex(s.index)["pred"])):
            if p is None:
                continue
            p = np.asarray(p, float)
            t.append({"set": ds, "system": n, "recall_hateful": 100 * p[hate].mean(), "fpr_other": 100 * p[~hate].mean(),
                      "n_hateful": int(hate.sum()), "n_other": int((~hate).sum())})
    t = pd.DataFrame(t)
    t.to_csv(out / "templates.csv", index=False)
    md += ["## Templates: political targets (cf) vs generic targets (gen)", "", t.round(1).to_markdown(index=False), ""]
    return md


def cv_summary(a, out):
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    gpu = Path(__file__).parent / "rev2_gpu" / "outputs"
    runs = sorted(gpu.glob("cv*_seed*/test.csv"))
    if not runs:
        return ["## Cross-validation over posts: not available yet", ""]
    d = rev2_data.phs_post().merge(pd.read_csv(config.REV2_DIR / "post_cv_folds.csv"), on="post_id")
    d["key"] = d["id"].astype(str)
    js = [clf(a.jev, s) for s in ("ptrain", "pval", "ptest") if clf(a.jev, s) is not None]
    jev = pd.concat(js) if len(js) == 3 else pd.DataFrame()
    mb = pd.concat([pd.read_csv(f).assign(seed=f.parent.name.split("_seed")[1], fold=f.parent.name.split("_")[0][2:]) for f in runs])
    mb["key"] = mb["id"].astype(str)
    rows = []
    for k in sorted(d.cv_fold.unique()):
        te, tr = d[d.cv_fold == k], d[d.cv_fold != k]
        lr = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000, C=10))
        lr.fit(tr["text"].fillna(""), tr["label"])
        r = {"fold": int(k), "posts": " ".join(map(str, sorted(te.post_id.unique()))), "n": len(te), "phs": int(te.label.sum()),
             "tfidf_lr": mf1(te.label, lr.predict(te["text"].fillna("")))}
        m = mb[mb.fold == str(k)]
        f1s = [mf1(g.label, g.pred) for _, g in m.groupby("seed")]
        if f1s:
            r["mbert_mean"], r["mbert_sd"] = np.mean(f1s), (np.std(f1s, ddof=1) if len(f1s) > 1 else np.nan)
        if len(jev):
            j = jev.reindex(te["key"])
            if j["pred"].notna().all():
                r["jev"] = mf1(te.label, j["pred"].astype(int))
                m1 = m[m.seed == m.seed.min()].set_index("key").reindex(te["key"])
                if m1["prob_phs"].notna().all():
                    r["mean_s1"] = mf1(te.label, ((m1["prob_phs"].values + j["prob_phs"].values) / 2 >= .5).astype(int))
                    r["disagreement_rate"] = 100 * (m1["pred"].values != j["pred"].values).mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "cv_over_posts.csv", index=False)
    return ["## Grouped 4-fold cross-validation over posts (macro-F1 per held-out fold)", "", t.round(2).to_markdown(index=False), ""]


def keep_hn(idx):
    
    f = config.REV2_DIR / "annotation_results.csv"
    if not f.exists():
        return np.ones(len(idx), bool)
    a = pd.read_csv(f).fillna("")
    a = a[a["item"].str.startswith("hn")]
    if ((a["A"] != "") & (a["B"] != "")).mean() < 0.9:     
        return np.ones(len(idx), bool)
    ok = set(a.loc[(a["A"] == "N") & (a["B"] == "N"), "item"].str[2:])
    return np.array([str(i) in ok for i in idx])


def accepted_pairs():
    
    f = config.REV2_DIR / "annotation_results.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f).fillna("")
    a = a[~a["item"].str.startswith("hn")]
    if ((a["A"] != "") & (a["B"] != "")).mean() < 0.9:     
        return None
    return set(a.loc[(a["A"] == "Y") & (a["B"] == "Y"), "item"].astype(str))


def flip_table(name, pred, cf2, acc):
    d = cf2.assign(pred=pd.Series(pred).reindex(cf2.index).values)
    is_orig = d["variant"].astype(str).isin(["0", "orig", "original"])
    rows = []
    rng = np.random.default_rng(rng_seed)
    for cls, lab in (("PHS originals", 1), ("non-HS originals", 0)):
        flips, accs = [], []
        for oid, g in d[d["label"] == lab].groupby("orig_id"):
            o = g[is_orig.reindex(g.index).values]
            v = g[~is_orig.reindex(g.index).values]
            if acc is not None:
                v = v[v.index.isin(acc)]
            if o.empty or v.empty or o["pred"].isna().any() or v["pred"].isna().any():
                continue
            flips.append(float((v["pred"] != o["pred"].iloc[0]).any()))
            accs.append(float(o["pred"].iloc[0] == lab))
        f = np.array(flips)
        bo = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(B)] if len(f) else [np.nan]
        rows.append({"system": name, "originals": cls, "n": len(f), "flip_rate": 100 * f.mean() if len(f) else np.nan,
                     "ci_lo": 100 * np.percentile(bo, 2.5), "ci_hi": 100 * np.percentile(bo, 97.5),
                     "accuracy_original": 100 * np.mean(accs) if accs else np.nan})
    return rows


if __name__ == "__main__":
    main()

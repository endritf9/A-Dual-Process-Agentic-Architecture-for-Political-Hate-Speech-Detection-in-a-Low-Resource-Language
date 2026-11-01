import numpy as np, pandas as pd
import config, rev2_data
from rev2_analysis import S1, clf, llm, cascade, route, mf1, LLMS
OUT = config.TABLES / "rev2"; md = []


def comp(y, a, b):
    ca, cb = a == y, b == y
    n11, n00, n10, n01 = (ca & cb).sum(), (~ca & ~cb).sum(), (ca & ~cb).sum(), (~ca & cb).sum()
    ea, eb = 1 - ca.mean(), 1 - cb.mean()
    return {"agree_%": 100 * (a == b).mean(), "acc_on_agree_%": 100 * n11 / max(1, n11 + n00), "DF_%": 100 * n00 / len(y),
            "DF_ind_%": 100 * ea * eb, "Q": (n11 * n00 - n01 * n10) / max(1, n11 * n00 + n01 * n10)}


st = S1("ptest", "mbert_post", "jev").frame(); y = st["y"].values
tf = clf("tfidf_lr", "ptest").reindex(st.index)["pred"].values
rows = [{"pair": "mBERT / Jev", **comp(y, st["yp_m"].values, st["yp_j"].values)},
        {"pair": "mBERT / TF-IDF LR", **comp(y, st["yp_m"].values, tf)}]
t = pd.DataFrame(rows); t.to_csv(OUT / "complementarity_ptest.csv", index=False)
md += ["## Complementarity on the test posts", "", t.round(2).to_markdown(index=False), ""]


rows = []
esc, base, _ = route(st, "disagree", 0)
for m in LLMS:
    a = llm("ptest", "agent_full", m)
    if a is None: continue
    s2 = a.reindex(st.index)["pred"].values
    ok = ~pd.isna(s2)
    p, _, c, _ = cascade(st, a, "disagree")
    rows.append({"llm": m, "esc_%": 100 * esc.mean(), "S1_acc_on_esc_%": 100 * (base[esc] == y[esc]).mean(),
                 "S2_acc_on_esc_%": 100 * (s2[esc & ok] == y[esc & ok]).mean(), "acc_S1_all_%": 100 * (base == y).mean(),
                 "acc_cascade_%": 100 * (p == y).mean(),
                 "oracle_rate_%": 100 * ((base != y) & ok & (s2 == y)).mean(),
                 "oracle_f1": mf1(y, np.where((base != y) & ok & (s2 == y), s2, base))})
t = pd.DataFrame(rows); t.to_csv(OUT / "decomposition_ptest.csv", index=False)
md += ["## Accuracy decomposition (P1) and oracle router", "", t.round(2).to_markdown(index=False), ""]


def tmpl(ds, name, pred, prob=None):
    d = rev2_data.load(ds)[0].assign(key=lambda x: x["cf_id"].astype(str)).set_index("key")
    d["pred"] = pd.Series(pred).reindex(d.index).values
    r = {"set": ds, "system": name}
    for g in sorted(d.group.unique()):
        s = d[d.group == g]["pred"].astype(float)
        r[("recall_" if "hate" in g else "FPR_") + g] = 100 * s.mean()
    if "template_id" in d.columns:
        fr = d.groupby("template_id")["pred"].nunique().gt(1).mean()
        r["flip_rate_templates_%"] = 100 * fr
        if prob is not None:
            d["prob"] = pd.Series(prob).reindex(d.index).values
            r["mean_prob_range"] = d.groupby("template_id")["prob"].agg(lambda v: v.max() - v.min()).mean()
    return r
rows = []
for ds in ("cf", "gen"):
    s = S1(ds, "mbert_post", "jev").frame()
    rows.append(tmpl(ds, "mBERT", s["yp_m"], s["pm"]))
    h = clf("mbert_post_hn", ds).reindex(s.index)
    rows.append(tmpl(ds, "mBERT trained with hard negatives", h["pred"], h["prob_phs"]))
    rows.append(tmpl(ds, "Jev", s["yp_j"], s["pj"]))
    rows.append(tmpl(ds, "Mean S1", s["yp_mean"], s["pmean"]))
    for cfg in ("zero_shot", "agent_full", "single_all", "agent_no_classifier"):
        r = llm(ds, cfg, "qwen3.7-flash")
        if r is not None and len(r) >= 0.95 * len(s):
            rows.append(tmpl(ds, f"Qwen 3.7 {cfg}", r.reindex(s.index)["pred"]))
            if cfg in ("agent_full", "single_all"):
                p, e, _, _ = cascade(s, r, "disagree")
                rows.append(tmpl(ds, f"cascade P1 + Qwen 3.7 {cfg} (esc {100*e.mean():.1f}%)", pd.Series(p, index=s.index)))
t = pd.DataFrame(rows); t.to_csv(OUT / "templates_full.csv", index=False)
md += ["## Template tests", "", t.round(1).to_markdown(index=False), ""]


a = pd.read_csv(config.REV2_DIR / "annotation_results.csv").fillna("")
for lab, part in (("pairs", a[~a["item"].str.startswith("hn")]), ("hard negatives", a[a["item"].str.startswith("hn")])):
    cats = sorted(set(part.A) | set(part.B)); pa = (part.A == part.B).mean()
    pi = {c: ((part.A == c).mean() + (part.B == c).mean()) / 2 for c in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / max(1, len(cats) - 1)
    md += [f"{lab}: n={len(part)}, raw agreement {100*pa:.1f}%, Gwet AC1 {(pa-pe)/(1-pe):.3f}, "
           f"A counts {part.A.value_counts().to_dict()}, B counts {part.B.value_counts().to_dict()}, both-accepted "
           f"{((part.A=='Y')&(part.B=='Y')).sum() if lab=='pairs' else ((part.A=='N')&(part.B=='N')).sum()}", ""]
(OUT / "extra.txt").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

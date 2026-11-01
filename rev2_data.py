import pandas as pd

import config

D = config.REV2_DIR


def phs_post():
    d = pd.read_csv(D / "phs_post.csv", encoding="utf-8")
    d["text"] = d["text"].fillna("").astype(str)
    return d


def load(ds):
    
    if ds in ("pval", "ptest", "ptrain"):
        d = phs_post()
        return d[d["split"] == {"pval": "val", "ptest": "test", "ptrain": "train"}[ds]].copy(), "id"
    if ds == "hn":
        d = pd.read_csv(D / "hard_negatives.csv", encoding="utf-8")
        return d[d["split"] != "train"].copy(), "id"
    if ds == "hn_train":
        d = pd.read_csv(D / "hard_negatives.csv", encoding="utf-8")
        return d[d["split"] == "train"].copy(), "id"
    if ds == "ncf2":
        return pd.read_csv(D / "natural_cf2.csv", encoding="utf-8"), "cf_id"
    if ds == "gen":
        return pd.read_csv(D / "generic_templates.csv", encoding="utf-8"), "cf_id"
    if ds == "cf":
        return pd.read_csv(config.COUNTERFACTUAL_CSV, encoding="utf-8"), "cf_id"
    raise ValueError(ds)


EVAL_SETS = ["pval", "ptest", "hn", "ncf2", "gen", "cf"]

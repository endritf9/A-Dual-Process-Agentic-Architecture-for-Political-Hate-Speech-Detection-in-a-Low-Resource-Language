import json
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import config
from common import ensure_dirs, get_logger, normalize
from counterfactual_templates import build_counterfactual_set

log = get_logger("01_prepare_data")


def find_raw_csv():
    for p in config.RAW_CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Raw CSV not found. Put it one folder above this package or at data/raw.csv. Tried: "
        + ", ".join(str(p) for p in config.RAW_CSV_CANDIDATES)
    )


def load_raw() -> pd.DataFrame:
    path = find_raw_csv()
    log.info(f"Reading {path}")
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
    df["Comment"] = df["Comment"].fillna("").astype(str)
    return df


def thread_context(df: pd.DataFrame) -> pd.DataFrame:
    

    comments = df["Comment"].tolist()
    post_key = [str(int(v)) if pd.notna(v) else None for v in df["Post"].tolist()]
    reply_root = [str(v).split("-")[0] if pd.notna(v) else None for v in df["Reply"].tolist()]
    post_text = {k: c for k, c in zip(post_key, comments) if k is not None}
    replies_by_root = {}
    for i, root in enumerate(reply_root):
        if root is not None:
            replies_by_root.setdefault(root, []).append(i)

    parent, related, is_reply = [], [], []
    for i in range(len(comments)):
        root = reply_root[i]
        if root is not None:
            is_reply.append(1)
            parent.append(post_text.get(root, ""))
            sib = [comments[j] for j in replies_by_root.get(root, []) if j != i]
            related.append(json.dumps(sib[:3], ensure_ascii=False))
        else:
            is_reply.append(0)
            parent.append("")
            kids = [comments[j] for j in replies_by_root.get(post_key[i], [])]
            related.append(json.dumps(kids[:3], ensure_ascii=False))
    df = df.copy()
    df["is_reply"], df["parent_text"], df["related_texts"] = is_reply, parent, related
    return df


def build_balanced(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["Class"].isin([config.NON_HS_CLASS, config.PHS_CLASS])]
    c0 = d[d["Class"] == config.NON_HS_CLASS].sample(n=config.N_NON_HS, random_state=config.SAMPLE_SEED)
    c1 = d[d["Class"] == config.PHS_CLASS]
    b = pd.concat([c0, c1])
    b["label"] = (b["Class"] == config.PHS_CLASS).astype(int)
    return b


def assign_folds(b: pd.DataFrame) -> pd.DataFrame:
    groups = b["text"].map(normalize)
    sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.FOLD_SEED)
    b = b.copy()
    b["fold"] = -1
    for k, (_, test_idx) in enumerate(sgkf.split(b, b["label"], groups)):
        b.iloc[test_idx, b.columns.get_loc("fold")] = k
    return b


def main():
    ensure_dirs(config.DATA_DIR, config.GPU_PACKAGE_DATA)
    raw = thread_context(load_raw())
    log.info(f"Raw rows: {len(raw)}; class counts: {raw['Class'].value_counts().to_dict()}")

    b = build_balanced(raw)
    b = b.rename(columns={"NrRen": "id", "Comment": "text"})
    b = assign_folds(b)
    
    
    _, orig_te = train_test_split(b, test_size=0.25, random_state=42)
    b["orig_test"] = b.index.isin(orig_te.index).astype(int)
    out = b[["id", "text", "label", "fold", "orig_test", "is_reply", "parent_text", "related_texts"]].sort_values("id")
    out.to_csv(config.FOLDS_CSV, index=False, encoding="utf-8")

    log.info(f"Balanced set: {len(out)} rows, labels {out['label'].value_counts().to_dict()}")
    log.info("Fold sizes / PHS share:\n" + out.groupby("fold")["label"].agg(["count", "mean"]).to_string())
    log.info(f"Replies in balanced set: {out['is_reply'].sum()}")
    o = out[out.orig_test == 1]
    log.info(f"Original split test set: {len(o)} comments, labels {o['label'].value_counts().to_dict()} "
             f"(expected 432 = 258 non-HS + 174 PHS)")
    dup = out["text"].map(normalize).duplicated().sum()
    log.info(f"Duplicate normalised texts kept in the same fold: {dup}")

    cf = build_counterfactual_set()
    cf.to_csv(config.COUNTERFACTUAL_CSV, index=False, encoding="utf-8")
    log.info(f"Counterfactual set: {len(cf)} rows; groups {cf['group'].value_counts().to_dict()}")

    for f in (config.FOLDS_CSV, config.COUNTERFACTUAL_CSV):
        shutil.copy(f, config.GPU_PACKAGE_DATA / f.name)
    log.info(f"Copied data files into {config.GPU_PACKAGE_DATA}")


if __name__ == "__main__":
    main()

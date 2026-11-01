import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

import config


def get_logger(name: str) -> logging.Logger:
    
    config.LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(config.LOGS / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    
    t = str(text).lower().replace("ë", "e").replace("ç", "c")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def binary_metrics(y_true, y_pred) -> dict:
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {"n": int(len(y_true)), "accuracy": 100 * accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"{avg}_p"], out[f"{avg}_r"], out[f"{avg}_f1"] = 100 * p, 100 * r, 100 * f
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    out["phs_p"], out["phs_r"], out["phs_f1"] = 100 * p, 100 * r, 100 * f
    return out


def split_masks(df, key):
    

    if str(key) == "orig":
        test = df["orig_test"] == 1
    else:
        test = df["fold"] == int(key)
    return ~test, test


def parse_split(v):
    return "orig" if str(v) == "orig" else int(v)


def ensure_dirs(*paths: Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

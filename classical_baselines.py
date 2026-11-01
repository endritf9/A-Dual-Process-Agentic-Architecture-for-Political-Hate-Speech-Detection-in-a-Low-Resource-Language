import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config
from common import binary_metrics, ensure_dirs, get_logger, split_masks

log = get_logger("02_classical_baselines")


def models():
    tfidf = lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    return {
        "tfidf_nb": make_pipeline(tfidf(), MultinomialNB()),
        "tfidf_svm": make_pipeline(tfidf(), LinearSVC(C=1.0)),
        "tfidf_rf": make_pipeline(tfidf(), RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
        "tfidf_lr": make_pipeline(tfidf(), LogisticRegression(max_iter=2000, C=10)),
    }


def proba(pipe, texts):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    d = pipe.decision_function(texts)
    return 1 / (1 + np.exp(-d))


def main():
    ensure_dirs(config.CLASSIFIER_PREDS, config.TABLES)
    df = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    cf = pd.read_csv(config.COUNTERFACTUAL_CSV, encoding="utf-8")
    rows = []
    for name in models():
        out = config.CLASSIFIER_PREDS / name
        ensure_dirs(out)
        for k in list(range(config.N_FOLDS)) + ["orig"]:
            pipe = models()[name]
            trm, tem = split_masks(df, k)
            tr, te = df[trm], df[tem]
            t0 = time.perf_counter()
            pipe.fit(tr["text"], tr["label"])
            p = proba(pipe, te["text"])
            pred = pipe.predict(te["text"])
            ms = 1000 * (time.perf_counter() - t0) / max(1, len(te))
            pd.DataFrame({"id": te["id"], "label": te["label"], "prob_phs": p, "pred": pred}) \
                .to_csv(out / f"fold{k}_test.csv", index=False)
            pc = proba(pipe, cf["text"])
            pd.DataFrame({"cf_id": cf["cf_id"], "label": cf["label"], "group": cf["group"], "prob_phs": pc,
                          "pred": pipe.predict(cf["text"])}).to_csv(out / f"fold{k}_cf.csv", index=False)
            m = binary_metrics(te["label"], pred)
            m.update({"model": name, "fold": k})
            rows.append(m)
            (out / f"fold{k}_metrics.json").write_text(json.dumps(m, indent=2))
        f1s = [r["macro_f1"] for r in rows if r["model"] == name and r["fold"] != "orig"]
        log.info(f"{name}: macro-F1 {np.mean(f1s):.2f} ± {np.std(f1s, ddof=1):.2f}")
    pd.DataFrame(rows).to_csv(config.TABLES / "classical_per_fold.csv", index=False)
    log.info("Done. Per-fold metrics in results/tables/classical_per_fold.csv")


if __name__ == "__main__":
    main()

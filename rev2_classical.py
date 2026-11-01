
import pandas as pd

import config
import rev2_data
from classical_baselines import models, proba
from common import binary_metrics, ensure_dirs, get_logger

log = get_logger("rev2_classical")


def main():
    ensure_dirs(config.CLASSIFIER_PREDS, config.TABLES)
    tr, _ = rev2_data.load("ptrain")
    rows = []
    for name, pipe in models().items():
        out = config.CLASSIFIER_PREDS / name
        ensure_dirs(out)
        pipe.fit(tr["text"], tr["label"])
        for ds in rev2_data.EVAL_SETS:
            d, key = rev2_data.load(ds)
            texts = d["text"].fillna("").astype(str)
            o = pd.DataFrame({key: d[key], "label": d["label"], "prob_phs": proba(pipe, texts), "pred": pipe.predict(texts)})
            o.to_csv(out / f"foldpost_{ds}.csv", index=False)
            if ds in ("pval", "ptest"):
                m = binary_metrics(o["label"], o["pred"])
                m.update({"model": name, "set": ds})
                rows.append(m)
                log.info(f"{name} {ds}: macro-F1 {m['macro_f1']:.2f}")
    pd.DataFrame(rows).to_csv(config.TABLES / "rev2_classical.csv", index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()

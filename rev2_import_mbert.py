import json
import shutil
from pathlib import Path

import pandas as pd

import config
from common import binary_metrics, ensure_dirs, get_logger

log = get_logger("rev2_import_mbert")
SRC = Path(__file__).parent / "rev2_gpu" / "outputs"
NAMES = {"val": "pval", "test": "ptest", "hn": "hn", "ncf2": "ncf2", "cf": "cf", "gen": "gen"}
TARGET = {"std": "mbert_post", "hn": "mbert_post_hn"}


def output_path(run, variant, filename):
    p = run / f"{filename}.csv"
    if p.exists():
        return p
    target = config.CLASSIFIER_PREDS / TARGET[variant] / f"foldpost_{NAMES[filename]}.csv"
    selected = config.CLASSIFIER_PREDS / TARGET[variant] / "selected_seed.json"
    if selected.exists() and json.loads(selected.read_text())["seed"] == int(run.name.split("_seed")[1]) and target.exists():
        return target
    raise FileNotFoundError(p)


def main():
    rows = []
    for run in sorted(list(SRC.glob("std_seed*")) + list(SRC.glob("hn_seed*"))):
        if run.name.startswith("smoke") or not (run / "done.json").exists():
            continue
        variant, seed = run.name.split("_seed")
        r = {"variant": variant, "seed": int(seed), "best_epoch": (run / "best_epoch.txt").read_text().strip()}
        for f, ds in (("val", "pval"), ("test", "ptest")):
            d = pd.read_csv(output_path(run, variant, f))
            m = binary_metrics(d["label"], d["pred"])
            r[f"{ds}_macro_f1"], r[f"{ds}_phs_r"] = m["macro_f1"], m["phs_r"]
        hn = pd.read_csv(output_path(run, variant, "hn"))
        r["hn_fpr"] = 100 * hn["pred"].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit(f"No finished runs in {SRC}")
    ensure_dirs(config.TABLES)
    t.to_csv(config.TABLES / "rev2_mbert_seeds.csv", index=False)
    log.info("\n" + t.to_string(index=False))
    log.info("\n" + t.groupby("variant")[["pval_macro_f1", "ptest_macro_f1", "hn_fpr"]].agg(["mean", "std"]).round(2).to_string())
    for variant, name in TARGET.items():
        tv = t[t["variant"] == variant]
        if tv.empty:
            continue
        best = tv.sort_values(["pval_macro_f1", "seed"], ascending=[False, True]).iloc[0]
        src = SRC / f"{variant}_seed{int(best['seed'])}"
        out = config.CLASSIFIER_PREDS / name
        ensure_dirs(out)
        for f, ds in NAMES.items():
            source = output_path(src, variant, f)
            destination = out / f"foldpost_{ds}.csv"
            if source.resolve() != destination.resolve():
                shutil.copy(source, destination)
        (out / "selected_seed.json").write_text(json.dumps({"seed": int(best["seed"]), "val_macro_f1": best["pval_macro_f1"]}))
        log.info(f"{name}: seed {int(best['seed'])} selected on validation (val macro-F1 {best['pval_macro_f1']:.2f})")


if __name__ == "__main__":
    main()

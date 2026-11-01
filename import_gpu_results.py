import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

import config
from common import ensure_dirs, get_logger

log = get_logger("07_import_gpu_results")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    zpath = Path(sys.argv[1])
    ensure_dirs(config.CLASSIFIER_PREDS, config.TABLES)
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zpath).extractall(tmp)
        src = Path(tmp) / "gpu_results"
        if not src.exists():
            src = Path(tmp)
        folds = pd.read_csv(config.FOLDS_CSV)
        for mdir in sorted(p for p in src.iterdir() if p.is_dir()):
            dst = config.CLASSIFIER_PREDS / mdir.name
            ensure_dirs(dst)
            n = 0
            for f in mdir.glob("fold*"):
                shutil.copy(f, dst / f.name)
                n += 1

            for t in dst.glob("fold*_test.csv"):
                k = int(t.stem.split("_")[0].replace("fold", ""))
                ids = set(pd.read_csv(t)["id"])
                expected = set(folds.loc[folds.fold == k, "id"])
                if ids != expected:
                    log.warning(f"{mdir.name} {t.name}: ids differ from data/phs_folds.csv "
                                f"({len(ids & expected)}/{len(expected)} match) - was the job run on the right data?")
            log.info(f"Imported {n} files for model '{mdir.name}'")
        for extra in ("latency.json", "env_info.json", "summary.csv", "gpu_run.log"):
            if (src / extra).exists():
                shutil.copy(src / extra, config.TABLES / f"gpu_{extra}")
        if (src / "latency.json").exists():
            log.info(f"Latency: {json.loads((src / 'latency.json').read_text())}")
    log.info("GPU results imported.")


if __name__ == "__main__":
    main()

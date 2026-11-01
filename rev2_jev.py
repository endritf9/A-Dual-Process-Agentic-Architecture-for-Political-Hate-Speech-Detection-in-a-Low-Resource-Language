import config
import jev_client as J
from common import ensure_dirs, get_logger
from rev2_data import EVAL_SETS, load
from run_jev import run_set

log = get_logger("rev2_jev")

if __name__ == "__main__":
    q = J.phs_questions()
    out = config.CLASSIFIER_PREDS / "jev"
    ensure_dirs(out)
    for ds in EVAL_SETS + ["ptrain"]:
        d, key = load(ds)
        cols = [key, "text", "label"] + (["group"] if "group" in d.columns else [])
        run_set("jev", q, d[cols].to_dict("records"), out / f"foldpost_{ds}.csv", key)
    log.info("Jev done on all revision-2 sets")

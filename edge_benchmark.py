import json
import os
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import torch

import config
from common import binary_metrics, ensure_dirs, get_logger, split_masks
from mbert_existing import load_model, predict

log = get_logger("edge_benchmark")
try:
    import psutil
except ImportError:
    psutil = None


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20 if psutil else float("nan")


def size_mb(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        path = f.name
    torch.save(model.state_dict(), path)
    s = os.path.getsize(path) / 2 ** 20
    os.remove(path)
    return s


def latency(model, tok, texts, max_len, threads):
    torch.set_num_threads(threads)
    predict(model, tok, texts[:5], max_len, bs=1)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        predict(model, tok, [t], max_len, bs=1)
        times.append(1000 * (time.perf_counter() - t0))
    return float(np.median(times)), float(np.percentile(times, 95)), float(np.mean(times))


def bench(name, model, tok, te, max_len, ref_pred=None, mem_before=None):
    texts = te["text"].tolist()
    peak = rss_mb()
    n_all = os.cpu_count() or 1
    res = {"model": name, "weights_mb": round(size_mb(model), 1),
           "rss_after_load_mb": round(rss_mb(), 0)}
    for th in (1, n_all):
        load_before = psutil.cpu_percent(interval=2) if psutil else float("nan")
        med, p95, mean = latency(model, tok, texts[:100], max_len, th)
        res[f"cpu_load_before_%_{th}thr"] = load_before
        res[f"latency_ms_median_{th}thr"] = round(med, 1)
        res[f"latency_ms_p95_{th}thr"] = round(p95, 1)
        peak = max(peak, rss_mb())
    torch.set_num_threads(n_all)
    t0 = time.perf_counter()
    p = predict(model, tok, texts, max_len, bs=16)
    dt = time.perf_counter() - t0
    peak = max(peak, rss_mb())
    pred = (p >= 0.5).astype(int)
    m = binary_metrics(te["label"], pred)
    res.update({"throughput_per_hour_bs16": round(3600 * len(texts) / dt), "peak_rss_mb": round(peak, 0),
                "macro_f1": round(m["macro_f1"], 2), "phs_f1": round(m["phs_f1"], 2)})
    if ref_pred is not None:
        res["agreement_with_fp32_%"] = round(100 * (pred == ref_pred).mean(), 1)
    log.info(json.dumps(res))
    return res, pred


def main():
    ensure_dirs(config.TABLES)
    df = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    te = df[split_masks(df, "orig")[1]].reset_index(drop=True)
    mem0 = rss_mb()
    t0 = time.perf_counter()
    model, tok, info = load_model()
    load_s = time.perf_counter() - t0
    max_len = int(info.get("max_length", 256))
    r32, pred32 = bench("mBERT FP32", model, tok, te, max_len)
    r32["load_time_s"] = round(load_s, 1)
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    r8, _ = bench("mBERT INT8 (dynamic)", q, tok, te, max_len, ref_pred=pred32)
    pd.DataFrame([r32, r8]).to_csv(config.TABLES / "T13_edge.csv", index=False)
    sysinfo = {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
               "cpu_count_logical": os.cpu_count(),
               "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else None,
               "ram_total_gb": round(psutil.virtual_memory().total / 2 ** 30, 1) if psutil else None,
               "torch": torch.__version__, "python": platform.python_version(), "rss_before_load_mb": round(mem0, 0)}
    (config.TABLES / "T13_edge_system.json").write_text(json.dumps(sysinfo, indent=2))
    log.info(f"System: {sysinfo}")
    log.info("Done: results/tables/T13_edge.csv")


if __name__ == "__main__":
    main()

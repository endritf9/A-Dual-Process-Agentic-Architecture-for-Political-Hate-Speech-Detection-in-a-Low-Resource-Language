import argparse
import json
import os
import platform
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "gpu_results"
LOG = HERE / "gpu_run.log"

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "distil_mbert": "distilbert-base-multilingual-cased",
}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def make_collate(tok):
    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        padded = tok.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return collate


@torch.no_grad()
def predict_proba(model, tok, texts, device, max_len, bs=32):
    model.eval()
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def metrics(y, p):
    out = {}
    for avg in ("macro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=avg, zero_division=0)
        out.update({f"{avg}_p": 100 * pr, f"{avg}_r": 100 * rc, f"{avg}_f1": 100 * f1})
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    out.update({"phs_p": 100 * pr, "phs_r": 100 * rc, "phs_f1": 100 * f1})
    return out


def run_one(name, hf_id, fold, df, cf, args, device):
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"fold{fold}_metrics.json").exists() and not args.force:
        log(f"[skip] {name} fold {fold} already done")
        return
    seed = 1000 + fold
    set_seed(seed)
    train = df[df["fold"] != fold]
    test = df[df["fold"] == fold]
    if args.smoke:
        train, test, cf = train.sample(64, random_state=0), test.sample(32, random_state=0), cf.head(16)

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2).to(device)
    ds = TextDS(train["text"], train["label"], tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log(f"[start] {name} fold {fold}: train={len(train)} test={len(test)} epochs={args.epochs} device={device}")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        log(f"  {name} fold {fold} epoch {ep + 1}/{args.epochs} loss={np.mean(losses):.4f}")
    train_time = time.time() - t0

    p_test = predict_proba(model, tok, test["text"].tolist(), device, args.max_len)
    pred = (p_test >= 0.5).astype(int)
    pd.DataFrame({"id": test["id"].values, "label": test["label"].values, "prob_phs": p_test, "pred": pred}) \
        .to_csv(out_dir / f"fold{fold}_test.csv", index=False)
    p_cf = predict_proba(model, tok, cf["text"].tolist(), device, args.max_len)
    pd.DataFrame({"cf_id": cf["cf_id"].values, "label": cf["label"].values, "group": cf["group"].values,
                  "prob_phs": p_cf, "pred": (p_cf >= 0.5).astype(int)}).to_csv(out_dir / f"fold{fold}_cf.csv", index=False)

    m = metrics(test["label"].values, pred)
    m.update({"model": name, "hf_id": hf_id, "fold": fold, "seed": seed, "train_seconds": round(train_time, 1),
              "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_len": args.max_len})
    (out_dir / f"fold{fold}_metrics.json").write_text(json.dumps(m, indent=2))
    log(f"[done] {name} fold {fold}: macro-F1={m['macro_f1']:.2f} weighted-F1={m['weighted_f1']:.2f} "
        f"PHS-F1={m['phs_f1']:.2f} ({train_time / 60:.1f} min)")

    if fold == 0 and name in args.latency_models:
        measure_latency(name, model, tok, test["text"].tolist(), device, args)
    if fold == 0 and name in args.save_models:
        save_dir = OUT / "models" / name
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        log(f"  saved {name} fold-0 model to {save_dir}")
    del model
    torch.cuda.empty_cache()


def measure_latency(name, model, tok, texts, device, args, n=100):
    
    res_path = OUT / "latency.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    sample = texts[:n]
    for dev in ([device] if device.type == "cpu" else [device, torch.device("cpu")]):
        model.to(dev)
        torch.set_num_threads(max(1, os.cpu_count() // 2)) if dev.type == "cpu" else None
        predict_proba(model, tok, sample[:5], dev, args.max_len, bs=1)
        t = time.perf_counter()
        predict_proba(model, tok, sample, dev, args.max_len, bs=1)
        ms = 1000 * (time.perf_counter() - t) / len(sample)
        res[f"{name}_{dev.type}_ms_per_comment"] = round(ms, 2)
        log(f"  latency {name} on {dev.type}: {ms:.1f} ms/comment")
    model.to(device)
    res_path.write_text(json.dumps(res, indent=2))


def env_info(device):
    info = {"host": socket.gethostname(), "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    import transformers
    info["transformers"] = transformers.__version__
    info["cpu"] = platform.processor()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--latency_models", nargs="*", default=["mbert", "distil_mbert"])
    ap.add_argument("--save_models", nargs="*", default=["mbert"],
                    help="models whose fold-0 weights are saved (about 700 MB for mBERT)")
    ap.add_argument("--smoke", action="store_true", help="tiny quick run to test the setup (1 epoch, few rows)")
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo runs that already exist")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not (args.allow_cpu or args.smoke):
        log("ERROR: no CUDA GPU found by PyTorch. Check GPU driver and CUDA-enabled PyTorch installation.")
        sys.exit(1)
    if args.smoke:
        global OUT
        OUT = HERE / "gpu_results_smoke"
        args.epochs = 1
        args.max_len = 64
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "phs_folds.csv", encoding="utf-8")
    cf = pd.read_csv(DATA / "counterfactual.csv", encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    info = env_info(device)
    (OUT / "env_info.json").write_text(json.dumps(info, indent=2))
    log(f"Environment: {info}")
    log(f"Data: {len(df)} comments, {df['fold'].nunique()} folds; counterfactual: {len(cf)}")

    for name in args.models:
        for fold in args.folds:
            try:
                run_one(name, MODELS[name], fold, df, cf, args, device)
            except torch.cuda.OutOfMemoryError:
                log(f"ERROR out of GPU memory in {name} fold {fold}. Re-run with --batch_size 4.")
                raise

    rows = [json.loads(p.read_text()) for p in OUT.glob("*/fold*_metrics.json")]
    if rows:
        s = pd.DataFrame(rows)
        s.to_csv(OUT / "summary.csv", index=False)
        agg = s.groupby("model")[["macro_f1", "weighted_f1", "phs_f1"]].agg(["mean", "std"]).round(2)
        log("Summary (mean/std over folds):\n" + agg.to_string())

    if not args.smoke:
        shutil.copy(LOG, OUT / "gpu_run.log")
        zip_base = HERE / f"gpu_results_{socket.gethostname()}"

        tmp = HERE / "_zip_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(OUT, tmp / "gpu_results", ignore=shutil.ignore_patterns("models"))
        shutil.make_archive(str(zip_base), "zip", tmp)
        shutil.rmtree(tmp)
        log(f"FINISHED: {zip_base}.zip")
    else:
        log("SMOKE TEST OK.")


if __name__ == "__main__":
    main()

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

import config
from common import binary_metrics, ensure_dirs, get_logger, split_masks

log = get_logger("09_mbert_existing")
NAME = "mbert_orig"


class CustomBert(nn.Module):
    

    def __init__(self, cfg, num_labels=2):
        super().__init__()
        self.model = BertModel(cfg, add_pooling_layer=True)
        self.classifier = nn.Sequential(nn.Linear(cfg.hidden_size, 256), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(256, num_labels))

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def find_state_dict(obj):
    if isinstance(obj, dict):
        if any(str(k).startswith("classifier.") for k in obj):
            return obj
        for v in obj.values():
            sd = find_state_dict(v)
            if sd is not None:
                return sd
    return None


def load_model():
    d = config.EXISTING_MBERT_DIR
    try:
        ckpt = torch.load(d / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(d / "model_checkpoint.pt", map_location="cpu", weights_only=False)
    sd = find_state_dict(ckpt)
    if sd is None:
        raise RuntimeError("No state dict with 'classifier.*' keys found in model_checkpoint.pt")
    cfg = BertConfig.from_json_file(str(d / "bert_model" / "config.json"))
    model = CustomBert(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    if missing or unexpected:
        log.warning(f"missing keys: {missing[:5]} ... unexpected keys: {list(unexpected)[:5]}")
    model.eval()
    tok = BertTokenizer.from_pretrained(str(d / "tokenizer"))
    info = json.loads((d / "model_info.json").read_text()) if (d / "model_info.json").exists() else {}
    if isinstance(ckpt, dict):
        meta = {k: v for k, v in ckpt.items() if isinstance(v, (int, float, str, list))}
        log.info(f"Checkpoint metadata: {meta}")
    return model, tok, info


@torch.no_grad()
def predict(model, tok, texts, max_len, bs=16):
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
    return np.concatenate(probs)


def run_ncf(model, tok, max_len, out):
    
    if not config.NATURAL_CF_CSV.exists():
        log.info("natural_cf.csv not found (run build_natural_cf.py first); skipped")
        return
    n = pd.read_csv(config.NATURAL_CF_CSV, encoding="utf-8")
    p = predict(model, tok, n["text"].fillna("").astype(str).tolist(), max_len)
    pd.DataFrame({"cf_id": n["cf_id"], "label": n["label"], "group": n["group"], "prob_phs": p,
                  "pred": (p >= 0.5).astype(int)}).to_csv(out / "foldorig_ncf.csv", index=False)
    log.info(f"Natural counterfactuals: {len(n)} sentences predicted -> {out / 'foldorig_ncf.csv'}")


def main():
    import sys
    if "--ncf-only" in sys.argv:
        out = config.CLASSIFIER_PREDS / NAME
        ensure_dirs(out)
        model, tok, info = load_model()
        run_ncf(model, tok, int(info.get("max_length", 256)), out)
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = config.CLASSIFIER_PREDS / NAME
    ensure_dirs(out, config.TABLES)
    model, tok, info = load_model()
    max_len = int(info.get("max_length", 256))
    df = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    te = df[split_masks(df, "orig")[1]]
    cf = pd.read_csv(config.COUNTERFACTUAL_CSV, encoding="utf-8")

    t0 = time.time()
    p = predict(model, tok, te["text"].tolist(), max_len)
    pred = (p >= 0.5).astype(int)
    pd.DataFrame({"id": te["id"], "label": te["label"], "prob_phs": p, "pred": pred}) \
        .to_csv(out / "foldorig_test.csv", index=False)
    m = binary_metrics(te["label"], pred)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(te["label"], pred).tolist()
    log.info(f"Original test split ({len(te)}): weighted-F1 {m['weighted_f1']:.2f}, macro-F1 {m['macro_f1']:.2f}, "
             f"PHS-F1 {m['phs_f1']:.2f}, confusion {cm} ({time.time() - t0:.0f} s)")
    if info.get("confusion_matrix"):
        same = cm == info["confusion_matrix"]
        log.info(f"Confusion matrix in model_info.json: {info['confusion_matrix']} -> "
                 f"{'MATCH: model and split reproduced exactly' if same else 'DIFFERENT - inspect model and split'}")
    (out / "foldorig_metrics.json").write_text(json.dumps({**m, "confusion": cm, "model": NAME}, indent=2))

    pc = predict(model, tok, cf["text"].tolist(), max_len)
    pd.DataFrame({"cf_id": cf["cf_id"], "label": cf["label"], "group": cf["group"], "prob_phs": pc,
                  "pred": (pc >= 0.5).astype(int)}).to_csv(out / "foldorig_cf.csv", index=False)
    for g in ("neutral", "critique", "place", "hateful"):
        s = (pc[cf["group"].values == g] >= 0.5).mean() * 100
        log.info(f"Counterfactual '{g}': {s:.1f}% predicted as PHS")
    run_ncf(model, tok, max_len, out)


    sample = te["text"].tolist()[:60]
    predict(model, tok, sample[:5], max_len, bs=1)
    t = time.perf_counter()
    predict(model, tok, sample, max_len, bs=1)
    ms = 1000 * (time.perf_counter() - t) / len(sample)
    lat_f = config.TABLES / "gpu_latency.json"
    lat = json.loads(lat_f.read_text()) if lat_f.exists() else {}
    lat[f"{NAME}_cpu_ms_per_comment"] = round(ms, 1)
    lat_f.write_text(json.dumps(lat, indent=2))
    log.info(f"CPU latency: {ms:.1f} ms per comment. Done.")


if __name__ == "__main__":
    main()

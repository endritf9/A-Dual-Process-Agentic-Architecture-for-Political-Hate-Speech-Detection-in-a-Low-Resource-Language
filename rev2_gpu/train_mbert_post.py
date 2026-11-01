import json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data" / "rev2", HERE / "outputs"
OUT.mkdir(exist_ok=True)
LOG = OUT / "server_log.txt"
MODEL = "bert-base-multilingual-cased"
EPOCHS, LR, BS, MAXLEN = 10, 1e-5, 8, 256
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = ["std", "hn"]


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " | " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class CustomBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL)
        self.classifier = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))

    def forward(self, ids, mask):
        return self.classifier(self.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])


def batches(texts, labels, tok, bs, shuffle, rng=None):
    idx = list(range(len(texts)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        enc = tok([texts[j] for j in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"], torch.tensor([labels[j] for j in b])


@torch.no_grad()
def predict(model, tok, texts, dev):
    model.eval()
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(list(texts[i:i + 32]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        lo = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        out.append(torch.softmax(lo, -1)[:, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    smoke = "--smoke" in sys.argv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pd.read_csv(DATA / "phs_post.csv"); d["text"] = d["text"].fillna("").astype(str)
    hn = pd.read_csv(DATA / "hard_negatives.csv"); hn["text"] = hn["text"].fillna("").astype(str)
    evals = {"val": d[d.split == "val"], "test": d[d.split == "test"],
             "hn": hn[hn.split != "train"], "ncf2": pd.read_csv(DATA / "natural_cf2.csv"),
             "cf": pd.read_csv(DATA.parent / "counterfactual.csv"), "gen": pd.read_csv(DATA / "generic_templates.csv")}
    tok = AutoTokenizer.from_pretrained(MODEL)
    for variant in VARIANTS:
        for seed in SEEDS:
            run = OUT / f"{variant}_seed{seed}"
            if (run / "done.json").exists():
                continue
            if (HERE / "STOP").exists():
                log("STOP file found, stopping"); return
            run.mkdir(exist_ok=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            tr = d[d.split == "train"][["text", "label"]]
            if variant == "hn":
                tr = pd.concat([tr, hn[hn.split == "train"][["text", "label"]]])
            if smoke:
                tr = tr.sample(64, random_state=seed)
            model = CustomBert().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            lossf = nn.CrossEntropyLoss()
            rng = random.Random(seed)
            best, hist, t0 = -1, [], time.time()
            texts, labels = tr["text"].tolist(), tr["label"].astype(int).tolist()
            for ep in range(1 if smoke else EPOCHS):
                model.train()
                for ids, mask, y in batches(texts, labels, tok, BS, True, rng):
                    opt.zero_grad()
                    loss = lossf(model(ids.to(dev), mask.to(dev)), y.to(dev))
                    loss.backward(); opt.step()
                pv = predict(model, tok, evals["val"]["text"].tolist(), dev)
                f1 = 100 * f1_score(evals["val"]["label"], (pv >= .5).astype(int), average="macro")
                hist.append({"epoch": ep + 1, "val_macro_f1": round(f1, 2)})
                log(f"{variant} seed {seed} epoch {ep + 1}: val macro-F1 {f1:.2f} ({time.time() - t0:.0f} s)")
                if f1 > best:
                    best = f1
                    for name, df in evals.items():
                        p = predict(model, tok, df["text"].fillna("").astype(str).tolist(), dev)
                        key = "cf_id" if "cf_id" in df.columns else "id"
                        pd.DataFrame({key: df[key].values, "label": df["label"].values, "prob_phs": p,
                                      "pred": (p >= .5).astype(int)}).to_csv(run / f"{name}.csv", index=False)
                    (run / "best_epoch.txt").write_text(str(ep + 1))
            (run / "done.json").write_text(json.dumps({"variant": variant, "seed": seed, "best_val_macro_f1": round(best, 2),
                                                        "history": hist, "train_size": len(tr),
                                                        "minutes": round((time.time() - t0) / 60, 1)}))
            log(f"DONE {variant} seed {seed}: best val macro-F1 {best:.2f}")
            del model, opt; torch.cuda.empty_cache()
            if smoke:
                log("smoke test finished"); return
    log("ALL RUNS FINISHED")


if __name__ == "__main__":
    main()

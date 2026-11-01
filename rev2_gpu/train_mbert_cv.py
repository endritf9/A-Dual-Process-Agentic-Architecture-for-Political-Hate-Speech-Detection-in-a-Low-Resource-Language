import json, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from train_mbert_post import CustomBert, batches, predict, log, DATA, OUT, LR, BS, HERE

SEEDS = [11, 22, 33]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = sorted(int((OUT / f"std_seed{s}" / "best_epoch.txt").read_text()) for s in (11, 22, 33, 44, 55))
    epochs = eps[len(eps) // 2]
    log(f"CV: fixed number of epochs = {epochs} (median of {eps})")
    d = pd.read_csv(DATA / "phs_post.csv"); d["text"] = d["text"].fillna("").astype(str)
    folds = pd.read_csv(DATA / "post_cv_folds.csv")
    d = d.merge(folds, on="post_id", how="left")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    for k in sorted(d.cv_fold.unique()):
        for seed in SEEDS:
            run = OUT / f"cv{k}_seed{seed}"
            if (run / "done.json").exists():
                continue
            if (HERE / "STOP").exists():
                log("STOP file found, stopping"); return
            run.mkdir(exist_ok=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            tr, te = d[d.cv_fold != k], d[d.cv_fold == k]
            model = CustomBert().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            lossf = nn.CrossEntropyLoss(); rng = random.Random(seed); t0 = time.time()
            texts, labels = tr["text"].tolist(), tr["label"].astype(int).tolist()
            for ep in range(epochs):
                model.train()
                for ids, mask, y in batches(texts, labels, tok, BS, True, rng):
                    opt.zero_grad(); loss = lossf(model(ids.to(dev), mask.to(dev)), y.to(dev)); loss.backward(); opt.step()
            p = predict(model, tok, te["text"].tolist(), dev)
            pd.DataFrame({"id": te["id"].values, "post_id": te["post_id"].values, "label": te["label"].values,
                          "prob_phs": p, "pred": (p >= .5).astype(int)}).to_csv(run / "test.csv", index=False)
            (run / "done.json").write_text(json.dumps({"fold": int(k), "seed": seed, "epochs": epochs, "n_test": len(te),
                                                        "minutes": round((time.time() - t0) / 60, 1)}))
            log(f"DONE cv fold {k} seed {seed} ({len(te)} test comments)")
            del model, opt; torch.cuda.empty_cache()
    log("ALL CV RUNS FINISHED")


if __name__ == "__main__":
    main()

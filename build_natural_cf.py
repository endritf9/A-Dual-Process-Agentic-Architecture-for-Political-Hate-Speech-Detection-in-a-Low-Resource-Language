import random
import re

import pandas as pd

import config
from common import get_logger, normalize, split_masks
from entity_kb import load_kb

log = get_logger("build_natural_cf")
N_SWAPS = 3
SEED = 7

POLITICIANS = {
    "Edi Rama": "Rama", "Sali Berisha": "Berisha", "Ilir Meta": "Meta", "Lulzim Basha": "Basha",
    "Erion Veliaj": "Veliaj", "Bajram Begaj": "Begaj", "Monika Kryemadhi": "Kryemadhi",
    "Albin Kurti": "Kurti", "Hashim Thaci": "Thaçi", "Vjosa Osmani": "Osmani", "Ramush Haradinaj": "Haradinaj",
    "Lumir Abdixhiku": "Abdixhiku", "Memli Krasniqi": "Krasniqi", "Kadri Veseli": "Veseli",
    "Fatmir Limaj": "Limaj", "Glauk Konjufca": "Konjufca", "Donika Gervalla": "Gervalla",
}
FULL = {k: k.replace("Thaci", "Thaçi") for k in POLITICIANS}
PARTIES = {"PS (Partia Socialiste)": "PS", "PD (Partia Demokratike)": "PD", "LSI": "LSI", "PDK": "PDK",
           "LDK": "LDK", "Vetevendosje (VV)": "Vetëvendosje", "AAK": "AAK", "Nisma": "Nisma"}
WORD = re.compile(r"\w+", re.UNICODE)


def spans_for(text, matched):
    
    toks = [(m.start(), m.end(), normalize(m.group())) for m in WORD.finditer(text)]
    toks = [t for t in toks if t[2]]
    target = matched.split()
    out = []
    for i in range(len(toks) - len(target) + 1):
        if [t[2] for t in toks[i:i + len(target)]] == target:
            out.append((toks[i][0], toks[i + len(target) - 1][1]))
    return out


def main():
    kb = load_kb()
    aliases = {e["name"]: {normalize(a) for a in e["aliases"]} for e in kb.entities}
    df = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    te = df[split_masks(df, "orig")[1]]
    rng = random.Random(SEED)
    rows, n_comments = [], 0
    for r in te.itertuples(index=False):
        ms = [m for m in kb.match(r.text) if (m["name"] in POLITICIANS or m["name"] in PARTIES)
              and m["matched"] in aliases[m["name"]]]
        ms = [m for m in ms if spans_for(r.text, m["matched"])]
        if not ms:
            continue
        m = ms[0]
        is_pol = m["name"] in POLITICIANS
        pool = POLITICIANS if is_pol else PARTIES
        present = {x["name"] for x in kb.match(r.text)}
        cands = sorted(n for n in pool if n not in present)
        targets = rng.sample(cands, N_SWAPS)
        full_mention = " " in m["matched"]
        n_comments += 1
        rows.append({"cf_id": f"n{r.id}_0", "orig_id": r.id, "variant": "orig", "text": r.text, "label": r.label,
                     "group": "natural", "entity_type": "politician" if is_pol else "party",
                     "source": m["name"], "target": m["name"]})
        for k, t in enumerate(targets, 1):
            surface = (FULL[t] if full_mention else POLITICIANS[t]) if is_pol else PARTIES[t]
            new, last = [], 0
            for a, b in spans_for(r.text, m["matched"]):
                orig_tok = r.text[a:b]
                s = surface.upper() if orig_tok.isupper() and len(orig_tok) > 1 else (
                    surface if orig_tok[:1].isupper() else surface.lower())
                new.append(r.text[last:a] + s)
                last = b
            new.append(r.text[last:])
            rows.append({"cf_id": f"n{r.id}_{k}", "orig_id": r.id, "variant": f"swap{k}", "text": "".join(new),
                         "label": r.label, "group": "natural", "entity_type": "politician" if is_pol else "party",
                         "source": m["name"], "target": t})
    out = pd.DataFrame(rows)
    out.to_csv(config.DATA_DIR / "natural_cf.csv", index=False, encoding="utf-8")
    o = out[out.variant == "orig"]
    log.info(f"Natural counterfactuals: {n_comments} real test comments -> {len(out)} sentences "
             f"(labels of the originals: {o.label.value_counts().to_dict()}, "
             f"types: {o.entity_type.value_counts().to_dict()})")
    for x in out.head(8).itertuples():
        log.info(f"  {x.variant:6s} {x.text[:110]}")


if __name__ == "__main__":
    main()

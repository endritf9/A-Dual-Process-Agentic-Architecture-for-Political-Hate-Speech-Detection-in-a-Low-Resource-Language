import random

import pandas as pd

import config
from build_natural_cf import FULL, PARTIES, POLITICIANS, spans_for
from common import get_logger, normalize
from counterfactual_templates import HATEFUL, NEUTRAL
from entity_kb import load_kb
from prepare_data import load_raw, thread_context

log = get_logger("rev2_build_sets")
OUT = config.DATA_DIR / "rev2"
ORIG_FILE = config.ROOT / "first_round_real_reviews" / "Dataset 3 dhe 1 FINAL.csv"
CAPS = {"train": 300, "val": 100, "test": 200}
N_ORIG = 100
N_SWAPS = 2
GENERIC = ["ky njeri", "fqinji im", "shefi im", "ai tip"]


def post_ids():
    o = pd.read_csv(ORIG_FILE, usecols=["NrRen", "Post"]).sort_values("NrRen").reset_index(drop=True)
    pid, cur, last = [], 0, None
    for v in o["Post"]:
        if pd.notna(v):
            if last is not None and v < last:
                cur += 1
            last = v
        pid.append(cur)
    o["post_id"] = pid
    return dict(zip(o["NrRen"], o["post_id"]))


FIRST_NAMES = {"edi", "edvin", "sala", "saliu", "iliri", "lulzimi", "albin", "albini", "hashimi", "hashim", "vjosa",
               "ramush", "ramushi", "lumiri", "memli", "memliu", "glauku", "donika", "ramuti"}


def mentions_any(kb, aliases, text):
    
    out = []
    for m in kb.match(text):
        if not (m["name"] in POLITICIANS or m["name"] in PARTIES):
            continue
        stems = [a for a in aliases[m["name"]] if m["matched"].startswith(a)]
        stem = max(stems, key=len) if stems else m["matched"]
        if stem in FIRST_NAMES:
            continue
        out.append(m)
    return out


def mentions(kb, aliases, text):
    out = []
    for m in kb.match(text):
        if not (m["name"] in POLITICIANS or m["name"] in PARTIES):
            continue
        if m["matched"] in FIRST_NAMES or m["matched"] not in aliases[m["name"]]:
            continue
        sp = spans_for(text, m["matched"])
        if not sp or any(b < len(text) and text[b] == "-" for a, b in sp):
            continue
        out.append(m)
    return out


def swap(text, m, target, kb):
    is_pol = m["name"] in POLITICIANS
    full_mention = " " in m["matched"]
    surface = (FULL[target] if full_mention else POLITICIANS[target]) if is_pol else PARTIES[target]
    new, last = [], 0
    for a, b in spans_for(text, m["matched"]):
        tok = text[a:b]
        s = surface.upper() if tok.isupper() and len(tok) > 1 else (surface if tok[:1].isupper() else surface.lower())
        new.append(text[last:a] + s)
        last = b
    new.append(text[last:])
    return "".join(new)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kb = load_kb()
    aliases = {e["name"]: {normalize(a) for a in e["aliases"]} for e in kb.entities}
    pmap = post_ids()
    raw = thread_context(load_raw())
    raw["post_id"] = raw["NrRen"].map(pmap)
    f = pd.read_csv(config.FOLDS_CSV, encoding="utf-8")
    split = pd.read_csv(config.DATA_DIR / "post_split.csv")[["id", "split"]]
    f = f.merge(split, on="id").merge(raw[["NrRen", "post_id"]], left_on="id", right_on="NrRen").drop(columns="NrRen")
    f.to_csv(OUT / "phs_post.csv", index=False, encoding="utf-8")
    post_split = f.groupby("post_id")["split"].first().to_dict()
    log.info(f"Experimental set: {f.split.value_counts().to_dict()}; posts per split: "
             f"{pd.Series(post_split).value_counts().to_dict()}")


    used = set(f["id"])
    c0 = raw[(raw["Class"] == config.NON_HS_CLASS) & (~raw["NrRen"].isin(used))].copy()
    c0["split"] = c0["post_id"].map(post_split)
    c0 = c0[c0["split"].notna()]
    c0["text"] = c0["Comment"].astype(str)
    c0 = c0[c0["text"].str.split().str.len() >= 4]
    c0["ents"] = c0["text"].map(lambda t: [m["name"] for m in mentions_any(kb, aliases, t)])
    c0 = c0[c0["ents"].str.len() > 0]
    hn = []
    for sp, cap in CAPS.items():
        s = c0[c0["split"] == sp]
        s = s.sample(min(cap, len(s)), random_state=1)
        hn.append(s)
        log.info(f"Hard negatives {sp}: {len(s)} of {int((c0['split'] == sp).sum())} available")
    hn = pd.concat(hn)
    hn = hn.rename(columns={"NrRen": "id"})
    hn["label"] = 0
    hn[["id", "text", "label", "split", "post_id", "is_reply", "parent_text", "related_texts"]].to_csv(
        OUT / "hard_negatives.csv", index=False, encoding="utf-8")


    rng = random.Random(7)
    te_phs = f[(f["split"] == "test") & (f["label"] == 1)]
    te_neg = hn[hn["split"] == "test"]
    rows = []
    for src, df_, lab in (("phs", te_phs, 1), ("hard_negative", te_neg, 0)):
        cands = [(r.id, r.text) for r in df_.itertuples() if mentions(kb, aliases, r.text)]
        rng.shuffle(cands)
        for oid, text in sorted(cands[:N_ORIG]):
            m = mentions(kb, aliases, text)[0]
            pool = POLITICIANS if m["name"] in POLITICIANS else PARTIES
            present = {x["name"] for x in kb.match(text)}
            targets = rng.sample(sorted(n for n in pool if n not in present), N_SWAPS)
            et = "politician" if m["name"] in POLITICIANS else "party"
            rows.append({"cf_id": f"m{oid}_0", "orig_id": oid, "variant": "orig", "text": text, "label": lab,
                         "group": src, "entity_type": et, "source": m["name"], "target": m["name"]})
            for k, t in enumerate(targets, 1):
                rows.append({"cf_id": f"m{oid}_{k}", "orig_id": oid, "variant": f"swap{k}",
                             "text": swap(text, m, t, kb), "label": lab, "group": src, "entity_type": et,
                             "source": m["name"], "target": t})
    ncf = pd.DataFrame(rows)
    ncf.to_csv(OUT / "natural_cf2.csv", index=False, encoding="utf-8")
    o = ncf[ncf.variant == "orig"]
    log.info(f"Natural CF v2: {len(o)} originals ({o.group.value_counts().to_dict()}), {len(ncf)} sentences")


    g = []
    for grp, temps, lab in (("neutral_generic", NEUTRAL, 0), ("hateful_generic", HATEFUL, 1)):
        for ti, t in enumerate(temps):
            for gi, e in enumerate(GENERIC):
                s = t.replace("{E}", e)
                g.append({"cf_id": f"g{grp[0]}{ti}_{gi}", "text": s[0].upper() + s[1:], "label": lab, "group": grp})
    pd.DataFrame(g).to_csv(OUT / "generic_templates.csv", index=False, encoding="utf-8")
    log.info(f"Generic-target templates: {len(g)} sentences")


    pairs = []
    for oid, grp in ncf.groupby("orig_id"):
        orig = grp[grp.variant == "orig"].iloc[0]
        for r in grp[grp.variant != "orig"].itertuples():
            pairs.append({"pair_id": r.cf_id, "original_comment": orig.text, "variant_with_new_name": r.text,
                          "name_in_original": orig.source, "name_in_variant": r.target})
    pairs = pd.DataFrame(pairs).sample(frac=1, random_state=3).reset_index(drop=True)
    instr = pd.DataFrame({"Instructions (please read first)": [
        "Each row shows a real comment and a copy in which only one politician or party name was replaced.",
        "Question: does the replacement change whether the comment is POLITICAL HATE SPEECH?",
        "Write in column 'label_preserved':  Y = the label stays the same (both are political HS, or both are not)",
        "                                    N = the label changes (for example, the comment only makes sense for the original person, or it becomes / stops being hateful)",
        "                                    ? = cannot decide",
        "Judge the label only, not whether the new sentence is factually true. A comment about a scandal of one specific "
        "person that becomes meaningless for the new person should be marked N.",
        "Work alone; do not discuss with the other annotator until both sheets are finished.",
        "Optional: write a short note in 'comment' for unclear cases."]})
    for who in ("A", "B"):
        with pd.ExcelWriter(OUT / f"annotation_sheet_annotator_{who}.xlsx") as xw:
            instr.to_excel(xw, sheet_name="instructions", index=False)
            p = pairs.copy()
            p["label_preserved"] = ""
            p["comment"] = ""
            p.to_excel(xw, sheet_name="pairs", index=False)
    log.info(f"Annotation sheets: {len(pairs)} pairs -> data/rev2/annotation_sheet_annotator_A.xlsx / _B.xlsx")


if __name__ == "__main__":
    main()

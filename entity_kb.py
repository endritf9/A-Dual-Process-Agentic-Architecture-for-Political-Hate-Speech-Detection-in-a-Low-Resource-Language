import csv
import json
from functools import lru_cache

import config
from common import normalize

MIN_PREFIX = 4
MAX_SUFFIX = 4


class EntityKB:
    def __init__(self, entities):
        self.entities = entities
        self.single, self.multi = [], []
        for e in entities:
            excl = {normalize(x) for x in e.get("exclude", [])}
            for a in e["aliases"]:
                na = normalize(a)
                if not na:
                    continue
                (self.multi if " " in na else self.single).append((na, e, excl))
        self.multi.sort(key=lambda t: -len(t[0]))

    def match(self, text: str):
        
        norm = normalize(text)
        tokens = norm.split()
        found = {}
        padded = f" {norm} "
        for na, e, _ in self.multi:
            if f" {na} " in padded:
                found.setdefault(e["name"], (e, na))
        for tok in tokens:
            for na, e, excl in self.single:
                if any(tok.startswith(x) for x in excl):
                    continue
                if tok == na or (
                    len(na) >= MIN_PREFIX and tok.startswith(na) and len(tok) - len(na) <= MAX_SUFFIX
                ):
                    found.setdefault(e["name"], (e, tok))
        return [
            {"name": e["name"], "type": e["type"], "country": e["country"], "role": e["role"], "matched": m}
            for e, m in found.values()
        ]

    def lookup(self, term: str):
        
        return self.match(term)


@lru_cache(maxsize=1)
def load_kb() -> EntityKB:
    data = json.loads((config.KB_DIR / "political_entities.json").read_text(encoding="utf-8"))
    entities = list(data["entities"])
    extra = config.KB_DIR / "extra_entities.csv"
    if extra.exists():
        with open(extra, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("name") and row.get("aliases"):
                    entities.append({
                        "name": row["name"], "type": row.get("type", "other"),
                        "country": row.get("country", "-"), "role": row.get("role", ""),
                        "aliases": [a.strip() for a in row["aliases"].split("|") if a.strip()],
                    })
    return EntityKB(entities)


if __name__ == "__main__":
    kb = load_kb()
    for s in ["Vrases kriminel drogaxhi edvin rama", "Kti kurti ja ka hup krejt", "mafia e pdk-se 20 vite",
              "Kosova ka vende te bukura", "Ramazani filloi sot"]:
        print(s, "->", [m["name"] for m in kb.match(s)])

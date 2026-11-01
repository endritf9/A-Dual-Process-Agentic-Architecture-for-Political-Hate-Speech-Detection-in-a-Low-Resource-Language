import pandas as pd

ACTORS = {
    "politician": ["Edi Rama", "Sali Berisha", "Albin Kurti", "Hashim Thaçi", "Vjosa Osmani",
                   "Ramush Haradinaj", "Ilir Meta", "Lumir Abdixhiku", "Memli Krasniqi", "Bajram Begaj"],
    "party": ["PS", "PD", "Vetëvendosje", "PDK", "LDK", "AAK"],
    "institution": ["Qeveria", "Kuvendi", "Policia", "Prokuroria", "Kryeministri"],
}
PLACES = ["Kosova", "Shqipëria", "Prishtina", "Tirana"]

NEUTRAL = [
    "{E} mbajti sot një konferencë për media.",
    "Sot në lajme pashë që {E} takoi ambasadorin e BE-së.",
    "{E} ka deklaruar se zgjedhjet do të mbahen në tetor.",
    "A e di dikush në çfarë ore flet {E} sonte në televizion?",
    "{E} uroi qytetarët për festën e sotme.",
    "Lexova të gjithë intervistën që dha {E} dje.",
    "{E} vizitoi sot shkollën e re në fshatin tonë.",
    "Po pres të shoh çfarë do të thotë {E} për buxhetin e vitit të ardhshëm.",
    "{E} njoftoi se projekti i rrugës fillon muajin e ardhshëm.",
    "Edhe vitin e kaluar {E} ishte në këtë aktivitet.",
]
CRITIQUE = [
    "Nuk pajtohem me vendimin që mori {E} për taksat.",
    "Mendoj se {E} nuk e ka mbajtur premtimin për vendet e punës.",
    "{E} duhet të japë llogari për shpenzimet e këtij projekti.",
    "Nuk do ta votoj më {E}, sepse programi nuk më bindi.",
    "{E} ka bërë shumë gabime në këtë mandat, shpresoj të ndryshojë diçka.",
    "Qëndrimi që mbajti {E} në debat ishte i dobët dhe pa argumente.",
]
HATEFUL = [
    "{E} është një qen i ndyrë dhe duhet zhdukur.",
    "{E} është hajdut dhe kriminel, pisllëk njeriu.",
    "Mallkuar qofsh {E}, tradhtar i shitur.",
    "{E} është një mi që duhet shtypur.",
    "{E} është plehë dhe duhet të përfundojë në varr.",
]
PLACE_NEUTRAL = [
    "{E} ka shumë vende të bukura për pushime.",
    "{E} luajti shumë mirë në ndeshjen e djeshme.",
    "Uroj që {E} të ketë më shumë vende pune për të rinjtë.",
    "{E} është vendlindja ime dhe e dua shumë.",
]


def build_counterfactual_set() -> pd.DataFrame:
    rows = []
    for group, templates, label in (("neutral", NEUTRAL, 0), ("critique", CRITIQUE, 0), ("hateful", HATEFUL, 1)):
        for t_i, t in enumerate(templates):
            for etype, names in ACTORS.items():
                for name in names:
                    rows.append({"text": t.format(E=name), "label": label, "group": group,
                                 "entity": name, "entity_type": etype, "template_id": f"{group}_{t_i}"})
    for t_i, t in enumerate(PLACE_NEUTRAL):
        for name in PLACES:
            rows.append({"text": t.format(E=name), "label": 0, "group": "place",
                         "entity": name, "entity_type": "place", "template_id": f"place_{t_i}"})
    df = pd.DataFrame(rows)
    df.insert(0, "cf_id", [f"cf{i:04d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    d = build_counterfactual_set()
    print(d.groupby(["group", "entity_type"]).size())
    print(d.sample(8, random_state=0)[["text", "label"]].to_string())

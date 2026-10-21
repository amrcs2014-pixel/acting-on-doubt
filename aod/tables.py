"""Small LaTeX table writers that do not depend on experiment results."""
import json
import os

from .detectors import ROOT

TAB = os.path.join(ROOT, "paper", "tables")
NAME = {"ciciot2023": "CICIoT2023", "edgeiiot": "Edge-IIoTset", "nfton_v3": "NF-ToN-IoT-v3"}


def datasets_table():
    d = json.load(open(os.path.join(TAB, "datasets.json")))
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Dataset & Flows & Feat. & Fam. & Ben./att. & AUC$_{\mathrm{dev}}$ & AUC$_{\mathrm{gw}}$\\", r"\midrule"]
    for k, v in d.items():
        L.append(f"{NAME[k]} & {v['n']:,} & {v['n_feat']} & {v['n_fam']} & "
                 f"{v['test_benign']:,}/{v['test_att']:,} & {v['weak_auc']:.3f} & {v['strong_auc']:.3f}\\\\"
                 .replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "datasets.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    datasets_table()
    print(open(os.path.join(TAB, "datasets.tex")).read())

"""Compare classifiers WITHOUT the source shortcut: do the 3 CT Colonography cancers score higher than CT Colonography benign patients?

    python src/ctc_only_rank.py A=D:\\runs\\A_densenet\\preds.csv B=D:\\runs\\B_resnet\\preds.csv D=D:\\runs\\D_resnet\\preds.csv

Why: on the main test set every model reaches AUC 1.00 because malignant = StageII and benign = CT Colonography, so models cannot be
told apart.  The 3 held-out cancers come from the SAME source as the benign group, so ranking them against the CT Colonography benign
patients (validation and test) is a comparison the source cannot decide.  It is tiny (3 cancers), so report it as descriptive: the
scores, the rank of each cancer, and the exact AUC with its permutation-style meaning (3 x N pairs), never a significance claim.
Per patient score = same rule as evaluate.py (mean over a scan's top-k slabs, then over scans).
Output per model: the 3 cancer scores, the benign score median and maximum, the share of benign patients each cancer beats, and the AUC.
Needs: numpy pandas (evaluate.py in the same folder)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import patient_scores

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path/to/preds.csv, one per model")
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv")); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tables", default=str(ROOT / "reports/tables"))
    a = ap.parse_args(); cs = pd.read_csv(a.split)[["patient_id", "source", "label", "split"]]; rows = []
    for item in a.runs:
        name, path = item.split("=", 1); pr = pd.read_csv(path)
        ps = patient_scores(pr, "topk", a.k).merge(cs, on="patient_id")
        ctc = ps[ps.source == "CTC"]; can = ctc[ctc.split == "challenge"].score.values
        ben = ctc[(ctc.label == "benign") & ctc.split.isin(["val", "test"])].score.values
        if len(can) == 0 or len(ben) == 0: print(f"{name}: no challenge or benign CT Colonography patients in the file"); continue
        auc = float(np.mean([(c > b) + 0.5 * (c == b) for c in can for b in ben]))
        beats = [float(np.mean(ben < c) + 0.5 * np.mean(ben == c)) for c in can]
        rows.append(dict(model=name, n_cancer=len(can), n_benign=len(ben), cancer_scores=" ".join(f"{c:.3f}" for c in sorted(can)),
                         benign_median=float(np.median(ben)), benign_max=float(ben.max()), share_of_benign_beaten=" ".join(f"{b:.2f}" for b in sorted(beats)),
                         auc_cancer_vs_ctc_benign=auc))
    R = pd.DataFrame(rows); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print("3 CT Colonography cancers against CT Colonography benign patients (validation + test), same source:")
    print(R.round(3).to_string(index=False)); Path(a.tables).mkdir(parents=True, exist_ok=True); R.to_csv(Path(a.tables) / "ctc_only_rank.csv", index=False)
    print("AUC 0.5 = the cancers score like benign patients; 1.0 = all three outrank every benign patient.  3 cancers only: descriptive.")


if __name__ == "__main__":
    main()

"""Patient-level evaluation: benign vs malignant.   Same script for every model and for the baseline.

A) Evaluate a model's predictions
    python src/evaluate.py --pred preds_densenet121.csv --name densenet121
   preds CSV needs:  patient_id, score            (score = probability of MALIGNANT, 0..1)
   optional columns: series_uid, slab_idx          (slice-level rows are combined automatically)

B) Make + evaluate the intensity-only baseline (a simple model that only sees brightness summaries)
    python src/evaluate.py --baseline intensity --name intensity_lr

What it does (Paper section 3.4 / Chapter 4 plan)
  * Slice scores -> series score (mean of the top-k slices) -> patient score (mean over a patient's series,
    so prone + supine count once).  The patient is the unit of analysis.
  * Label and split come ONLY from the frozen cohort_split.csv, never from the predictions file.
  * The decision threshold is chosen on the VALIDATION patients (Youden's J) and then applied unchanged to test.
    Nothing is tuned on test.  Pass --threshold to fix it by hand.
  * Test: AUC with a bootstrap 95% CI (resampling patients within class), sensitivity and specificity with exact
    Clopper-Pearson 95% CIs, balanced accuracy, confusion matrix.
  * Challenge set (the 3 held-out CTC cancers): reported only as "detected x of 3" with an exact CI, no AUC.
Outputs: reports/tables/eval_<name>.csv, eval_<name>_patients.csv, reports/figures/eval_<name>_roc.png
Needs: pip install numpy pandas scipy scikit-learn matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261023
N_BOOT = 2000


def clopper_pearson(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def patient_scores(pred, agg, k):
    """slice/series rows -> one score per patient."""
    if "series_uid" not in pred: pred = pred.assign(series_uid=pred.patient_id)
    def top(x):
        x = np.sort(np.asarray(x, float))[::-1]
        return float(x[: max(k, 1)].mean()) if agg == "topk" else float(x.mean() if agg == "mean" else x.max())
    ser = pred.groupby(["patient_id", "series_uid"]).score.apply(top).reset_index()
    return ser.groupby("patient_id").score.mean().reset_index()


def boot_auc(y, s, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); out = []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s); j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def metrics(y, s, thr, rng):
    pred = (s >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    P, N = int((y == 1).sum()), int((y == 0).sum())
    sens, spec = tp / P, tn / N
    auc = roc_auc_score(y, s); ci = boot_auc(y, s, rng)
    se, sp = clopper_pearson(tp, P), clopper_pearson(tn, N)
    return dict(n=len(y), n_malignant=P, n_benign=N, auc=auc, auc_lo=ci[0], auc_hi=ci[1], threshold=thr,
                sensitivity=sens, sens_lo=se[0], sens_hi=se[1], specificity=spec, spec_lo=sp[0], spec_hi=sp[1],
                balanced_accuracy=(sens + spec) / 2, TP=tp, FN=P - tp, TN=tn, FP=N - tn)


def make_intensity_baseline(a):
    """Logistic regression on 4 intensity summaries per series, trained on TRAIN patients only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pre = pd.read_parquet(a.preproc); pre = pre[pre.ok == True]
    man = pd.read_csv(a.download)[["series_uid", "patient_id"]].drop_duplicates("series_uid")
    cs = pd.read_csv(a.split)[["patient_id", "label", "split"]]
    d = pre.merge(man, on="series_uid").merge(cs, on="patient_id")
    feats = ["pre_p1", "pre_p50", "pre_p99", "pre_air_frac"]
    tr = d[(d.split == "train") & d.label.isin(["benign", "malignant"])]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    m.fit(tr[feats], (tr.label == "malignant").astype(int))
    d["score"] = m.predict_proba(d[feats])[:, 1]
    return d[["patient_id", "series_uid", "score"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--name", required=True)
    ap.add_argument("--baseline", choices=["intensity"])
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--preproc", default=str(ROOT / "data/processed/preproc_series.parquet"))
    ap.add_argument("--download", default=str(ROOT / "data/manifests/download_series.csv"))
    ap.add_argument("--agg", choices=["topk", "mean", "max"], default="topk")
    ap.add_argument("--k", type=int, default=5, help="slices averaged per series when --agg topk")
    ap.add_argument("--threshold", type=float, help="fix the threshold by hand instead of choosing it on validation")
    ap.add_argument("--tables", default=str(ROOT / "reports/tables")); ap.add_argument("--figs", default=str(ROOT / "reports/figures"))
    a = ap.parse_args()
    if bool(a.pred) == bool(a.baseline): ap.error("give exactly one of --pred or --baseline")

    pred = make_intensity_baseline(a) if a.baseline else pd.read_csv(a.pred)
    if not {"patient_id", "score"} <= set(pred.columns): ap.error("predictions need columns patient_id, score")
    if pred.score.isna().any() or not pred.score.between(0, 1).all(): ap.error("scores must be probabilities in [0, 1] with no blanks")
    cs = pd.read_csv(a.split)[["patient_id", "label", "split"]]
    unknown = set(pred.patient_id) - set(cs.patient_id)
    if unknown: ap.error(f"{len(unknown)} patients in the predictions are not in cohort_split.csv, e.g. {sorted(unknown)[:3]}")
    ps = patient_scores(pred, a.agg, a.k).merge(cs, on="patient_id")
    ps["y"] = (ps.label == "malignant").astype(int)
    rng = np.random.default_rng(SEED)

    sets = {s: ps[(ps.split == s) & ps.label.isin(["benign", "malignant"])] for s in ["train", "val", "test"]}
    exp = cs[cs.split.isin(["val", "test"])].groupby("split").size()
    for s in ["val", "test"]:
        if len(sets[s]) < exp.get(s, 0): print(f"WARNING: only {len(sets[s])} of {exp[s]} {s} patients have predictions")
        if sets[s].y.nunique() < 2: ap.error(f"{s} set needs both classes to compute AUC")
    thr = a.threshold if a.threshold is not None else youden(sets["val"].y.values, sets["val"].score.values)

    rows = []
    for s in ["train", "val", "test"]:
        if len(sets[s]) and sets[s].y.nunique() == 2:
            r = metrics(sets[s].y.values, sets[s].score.values, thr, rng); r["set"] = s; rows.append(r)
    ch = ps[ps.split == "challenge"]
    if len(ch):
        det = int((ch.score >= thr).sum()); lo, hi = clopper_pearson(det, len(ch))
        rows.append(dict(set="challenge", n=len(ch), n_malignant=len(ch), threshold=thr, sensitivity=det / len(ch),
                         sens_lo=lo, sens_hi=hi, TP=det, FN=len(ch) - det))
    res = pd.DataFrame(rows).set_index("set")
    Path(a.tables).mkdir(parents=True, exist_ok=True); Path(a.figs).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(a.tables) / f"eval_{a.name}.csv")
    ps.drop(columns="y").to_csv(Path(a.tables) / f"eval_{a.name}_patients.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{a.name}: aggregation = {a.agg}{' (k=' + str(a.k) + ')' if a.agg == 'topk' else ''}, threshold = {thr:.3f}"
          f"{' (set by hand)' if a.threshold is not None else ' (Youden on validation)'}")
    show = res[["n", "auc", "auc_lo", "auc_hi", "sensitivity", "sens_lo", "sens_hi", "specificity", "spec_lo", "spec_hi", "TP", "FN", "TN", "FP"]]
    print(show.round(3).to_string())
    if "test" in res.index: print("\nTEST is the number to report. Train and val are for checking overfitting only.")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for s, c in [("val", "#8a8a8a"), ("test", "#2e86e4")]:
        if len(sets[s]) and sets[s].y.nunique() == 2:
            f, t, _ = roc_curve(sets[s].y, sets[s].score); ax.plot(f, t, color=c, lw=2, label=f"{s} (AUC {res.loc[s, 'auc']:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=1); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
    ax.set_title(f"{a.name}: patient-level ROC", fontsize=10); ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(Path(a.figs) / f"eval_{a.name}_roc.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

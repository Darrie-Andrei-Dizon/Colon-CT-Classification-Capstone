"""Within-CT-Colonography analysis: advanced adenoma vs smaller / non-neoplastic polyps, patient-level cross-validation.

Why this analysis: every scan here comes from ONE dataset with ONE preparation and protocol, so the StageII-vs-CTC
source shortcut does not apply.  It is exploratory: only 95 labelled patients, so intervals are wide.

    python src/within_ctc_cv.py --model size                       # polyp-size rule (reference, partly circular)
    python src/within_ctc_cv.py --model intensity --npy-dir D:\\tcia\\processed
    python src/within_ctc_cv.py --model embeddings --embeddings embeddings_densecl.parquet

Patients: the 95 labelled CTC patients in cohort_split.csv (train/val/test pooled; the frozen split is for the main
analysis).  The 3 CTC cancers (challenge set) and the excluded patients are NOT used.
Each patient gets ONE group from their worst lesion:
    advanced_adenoma     adenoma that is >= 10 mm, or tubulovillous / villous (codes 14, 16)
    nonadvanced_adenoma  tubular / adenomatous adenoma < 10 mm (codes 12, 13)
    nonneoplastic        hyperplastic or normal mucosa only (codes 10, 17)
Our table has no separate high-grade-dysplasia code, so this is an approximation of the usual definition.
Tasks (positive vs negative):
    adv_vs_nonneoplastic       advanced adenoma  vs  non-neoplastic            (28 vs 25)
    adv_vs_other_benign        advanced adenoma  vs  everything else           (28 vs 67)
    adenoma_vs_nonneoplastic   any adenoma       vs  non-neoplastic            (70 vs 25)

Models
    size         score = largest polyp size in the labels.  The advanced-adenoma label is partly DEFINED by size, so this
                 is a reference ceiling, not a fair competitor to an image model.
    intensity    4 brightness summaries from inside the colon outline, averaged over a patient's scans (logistic regression).
                 Should sit near 0.5 here; if it does not, something other than the lesion separates the groups.
    embeddings   a linear probe on patient-level features you supply (e.g. DenseCL backbone output averaged over a patient's
                 colon slabs): a parquet/csv with patient_id plus numeric columns.  With ~50-95 patients, train a small
                 linear model on frozen features rather than fine-tuning a whole network.

Cross-validation: stratified 5-fold, repeated 20 times with fixed seeds, always splitting by PATIENT.  Every patient gets
an out-of-fold score each repeat; the final score is the average over repeats.  Reported: AUC of those scores, a 95% CI from
a bootstrap over patients, the spread of AUC across repeats, and sensitivity / specificity at probability 0.5 with exact
Clopper-Pearson intervals.  Nothing is tuned on the held-out fold.
Note: if the features come from a network pretrained (DenseCL) on train-side patients, say so; those patients' images were seen
without labels.
Outputs: reports/tables/within_ctc_patients.csv, within_ctc_cv_<model>.csv, within_ctc_cv_<model>_oof.csv, reports/figures/within_ctc_cv_<model>.png
Needs: numpy pandas scipy scikit-learn matplotlib pyarrow
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
SEED = 20261023
N_BOOT = 2000
ORDER = ["cancer", "advanced_adenoma", "nonadvanced_adenoma", "nonneoplastic", "other"]
TASKS = {"adv_vs_nonneoplastic": (["advanced_adenoma"], ["nonneoplastic"]),
         "adv_vs_other_benign": (["advanced_adenoma"], ["nonadvanced_adenoma", "nonneoplastic"]),
         "adenoma_vs_nonneoplastic": (["advanced_adenoma", "nonadvanced_adenoma"], ["nonneoplastic"])}


def clopper_pearson(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan)
    return (0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)))


def patient_table(polyps_path, split_path):
    p = pd.read_parquet(polyps_path); cs = pd.read_csv(split_path)
    c = cs[(cs.source == "CTC") & cs.split.isin(["train", "val", "test"])][["patient_id", "split"]]
    p = p[p.patient_id.isin(c.patient_id)].copy()
    adenoma = p.histology_code.isin([12, 13, 14, 16]); villous = p.histology_code.isin([14, 16]); big = p.size_mm >= 10
    p["group"] = np.select([p.histology_code == 1, adenoma & (big | villous), adenoma, p.histology_code.isin([10, 17]),
                            p.histology_code == 88], ["cancer", "advanced_adenoma", "nonadvanced_adenoma", "nonneoplastic", "other"], "unknown")
    rank = {k: i for i, k in enumerate(ORDER)}
    w = p.assign(r=p.group.map(rank)).sort_values("r").groupby("patient_id").first()[["group"]]
    w["max_polyp_mm"] = p.groupby("patient_id").size_mm.max(); w["n_polyps"] = p.groupby("patient_id").size()
    return w.reset_index().merge(c, on="patient_id")


def intensity_features(pt, a):
    cache = Path(a.cache)
    if cache.exists() and not a.recompute: return pd.read_parquet(cache)
    from source_check_colon import body_mask, features
    sl = pd.read_csv(a.slices); sl = sl[sl.patient_id.isin(pt.patient_id)]; rows = []
    n = sl.series_uid.nunique()
    for k, (uid, g) in enumerate(sl.groupby("series_uid"), 1):
        z = np.sort(g.slab_idx.values)
        vol = np.load(Path(a.npy_dir) / f"{uid}.npy", mmap_mode="r")[z].astype(np.float32)
        m = np.load(Path(a.masks) / f"{uid}.npz")["mask"][z] & body_mask(vol)
        if m.sum() < 500: print(f"skip {uid[-10:]}: colon region too small"); continue
        f = features(vol[m]); rows.append(dict(patient_id=g.patient_id.iloc[0], **{c: f[c] for c in ["p1", "p50", "p99", "dark_frac"]}))
        if k % 25 == 0 or k == n: print(f"{k}/{n}", flush=True)
    d = pd.DataFrame(rows).groupby("patient_id").mean(numeric_only=True).reset_index()
    cache.parent.mkdir(parents=True, exist_ok=True); d.to_parquet(cache); return d


def cv_scores(X, y, C, reps, folds):
    oof = np.full((reps, len(y)), np.nan)
    for r in range(reps):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED + r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X[tr], y[tr])
            oof[r, te] = m.predict_proba(X[te])[:, 1]
    return oof


def perm_p(X, y, C, folds, n_perm, rng, s_fixed=None, reps=3):
    """Permutation p-value: how often shuffled labels give an AUC as high as the real one (same CV scheme, 3 repeats each)."""
    def stat(yy):
        s = s_fixed if s_fixed is not None else cv_scores(X, yy, C, reps, folds).mean(0)
        return roc_auc_score(yy, s)
    obs = stat(y); null = np.array([stat(rng.permutation(y)) for _ in range(n_perm)])
    return float((1 + (null >= obs).sum()) / (1 + n_perm))


def summarise(y, s, per_repeat_auc, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); boot = []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]; boot.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5]); pred = (s >= 0.5).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()); P, N = len(pos), len(neg)
    se, sp = clopper_pearson(tp, P), clopper_pearson(tn, N)
    return dict(n_pos=P, n_neg=N, auc=roc_auc_score(y, s), auc_lo=lo, auc_hi=hi,
                auc_repeat_mean=float(np.mean(per_repeat_auc)) if len(per_repeat_auc) else np.nan,
                auc_repeat_sd=float(np.std(per_repeat_auc)) if len(per_repeat_auc) else np.nan,
                sensitivity=tp / P, sens_lo=se[0], sens_hi=se[1], specificity=tn / N, spec_lo=sp[0], spec_hi=sp[1], TP=tp, FN=P - tp, TN=tn, FP=N - tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["size", "intensity", "embeddings"], required=True)
    ap.add_argument("--embeddings", help="parquet/csv: patient_id + numeric feature columns")
    ap.add_argument("--task", choices=list(TASKS), help="one task (default: all three)")
    ap.add_argument("--C", type=float, help="inverse regularisation (default 1.0 intensity, 0.05 embeddings)")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--perm", type=int, default=200, help="label shuffles for the permutation p-value (0 = skip)")
    ap.add_argument("--npy-dir"); ap.add_argument("--name")
    ap.add_argument("--polyps", default=str(ROOT / "data/processed/polyps.parquet"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--masks", default=str(ROOT / "data/processed/colon"))
    ap.add_argument("--cache", default=str(ROOT / "data/processed/within_ctc_intensity.parquet")); ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--tables", default=str(ROOT / "reports/tables")); ap.add_argument("--figs", default=str(ROOT / "reports/figures"))
    a = ap.parse_args(); name = a.name or a.model
    if a.model == "intensity" and not a.npy_dir: ap.error("--npy-dir is required for --model intensity")
    if a.model == "embeddings" and not a.embeddings: ap.error("--embeddings is required for --model embeddings")

    pt = patient_table(a.polyps, a.split); Path(a.tables).mkdir(parents=True, exist_ok=True); Path(a.figs).mkdir(parents=True, exist_ok=True)
    pt.to_csv(Path(a.tables) / "within_ctc_patients.csv", index=False)
    print("patients per group:", pt.group.value_counts().to_dict())
    if a.model == "size":
        print("NOTE: the advanced-adenoma label is partly defined by polyp size, so this is a reference ceiling, not a fair competitor.")
        feat, cols, C = pt[["patient_id"]].assign(size=pt.max_polyp_mm.values), ["size"], None
    elif a.model == "intensity":
        feat, cols, C = intensity_features(pt, a), ["p1", "p50", "p99", "dark_frac"], a.C or 1.0
    else:
        feat = pd.read_parquet(a.embeddings) if a.embeddings.endswith(".parquet") else pd.read_csv(a.embeddings)
        cols = [c for c in feat.columns if c != "patient_id" and pd.api.types.is_numeric_dtype(feat[c])]; C = a.C or 0.05
        print(f"embeddings: {len(cols)} features for {feat.patient_id.nunique()} patients")
    d = pt.merge(feat, on="patient_id", how="inner")
    if len(d) < len(pt): print(f"WARNING: features missing for {len(pt) - len(d)} of {len(pt)} patients (dropped)")

    rng = np.random.default_rng(SEED); out, oof_rows, curves = [], [], {}
    for task in ([a.task] if a.task else list(TASKS)):
        posg, negg = TASKS[task]; t = d[d.group.isin(posg + negg)].reset_index(drop=True)
        y = t.group.isin(posg).astype(int).values
        if min(y.sum(), (1 - y).sum()) < a.folds: print(f"skip {task}: too few patients in one class"); continue
        if a.model == "size":
            s = np.clip(t["size"].values / 20.0, 0, 1); rep = []                   # 10 mm maps to 0.5
        else:
            oof = cv_scores(t[cols].values.astype(float), y, C, a.repeats, a.folds)
            rep = [roc_auc_score(y, oof[r]) for r in range(a.repeats)]; s = oof.mean(0)
        r = summarise(y, s, rep, rng); r.update(task=task, model=name)
        if a.perm > 0:
            r["p_perm"] = perm_p(None if a.model == "size" else t[cols].values.astype(float), y, C, a.folds, a.perm, rng,
                                 s_fixed=s if a.model == "size" else None)
        out.append(r)
        oof_rows.append(pd.DataFrame(dict(task=task, patient_id=t.patient_id, group=t.group, y=y, score=s)))
        curves[task] = roc_curve(y, s)
    res = pd.DataFrame(out).set_index("task")
    res.to_csv(Path(a.tables) / f"within_ctc_cv_{name}.csv"); pd.concat(oof_rows).to_csv(Path(a.tables) / f"within_ctc_cv_{name}_oof.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n{name}: {'reference rule on all patients' if a.model == 'size' else f'{a.folds}-fold x {a.repeats} repeats, split by patient'} (exploratory)")
    print("auc_lo/auc_hi: bootstrap over patients (rough: too narrow for small samples).  p_perm: share of shuffled-label runs that did as well (the more trustworthy check).")
    show = ["n_pos", "n_neg", "auc", "auc_lo", "auc_hi", "auc_repeat_sd", "p_perm", "sensitivity", "sens_lo", "sens_hi", "specificity", "spec_lo", "spec_hi"]
    print(res[[c for c in show if c in res.columns]].round(3).to_string())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for (task, (f, tp, _)), c in zip(curves.items(), ["#2e86e4", "#e4572e", "#2ea36a"]):
        ax.plot(f, tp, color=c, lw=2, label=f"{task} (AUC {res.loc[task, 'auc']:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=1); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
    ax.set_title(f"Within CT Colonography: {name} (exploratory)", fontsize=10); ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(Path(a.figs) / f"within_ctc_cv_{name}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

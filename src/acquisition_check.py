"""Acquisition / site check (advisor question 9): do scan settings, scanner or patient details separate the labels?

    python src/acquisition_check.py [--slices data/processed/colon_slices.csv]

Two parts, both from the metadata tables (no images needed).
 Part 1  StageII vs CT Colonography: how different are the two sources in scanner, series length, slice thickness,
         reconstruction kernel, age and sex?  This documents the source shortcut (a model can use these differences).
 Part 2  Inside CT Colonography only (one dataset, one protocol): do acquisition details predict the labels of the three
         within-CTC tasks (advanced adenoma vs non-neoplastic, vs other benign, any adenoma vs non-neoplastic)?
         - association tests per feature (permutation chi-square for categories, Mann-Whitney for numbers)
         - a model that sees ONLY acquisition details (scanner model, kernel, thickness, slice count, age, sex, and the patient-id style, a probable site or batch marker: '1.3.6...' vs 'CTC-...'),
           same patient-level repeated 5-fold CV as within_ctc_cv.py, with AUC, bootstrap CI and permutation p
         - balance of scanner models across the frozen train / val / test split
 How to read Part 2: AUC near 0.5 and no clear association = acquisition does not explain the within-CTC result, so an
 image model that beats 0.5 there is using something else.  AUC clearly above 0.5 = a scanner or setting is linked to the
 label; report it and treat within-CTC image results with that caveat.

Which series count: the series in colon_slices.csv when the file exists (the scans actually used), otherwise every
non-localizer volume series.  Per patient: most common value for categories, median for numbers.
Series description examples: 'Colo_prone  1.0  B30f' -> thickness 1.0 mm, kernel B30f; 'Pelvis  5.0  B31f'.
Note: CTC study dates are removed in the public data, so scan year can be compared only inside StageII.
Outputs: reports/tables/acq_source_summary.csv, acq_within_ctc_features.csv, acq_within_ctc_assoc.csv,
         acq_within_ctc_model.csv, acq_split_balance.csv, reports/figures/acq_within_ctc.png
Needs: numpy pandas scipy scikit-learn matplotlib pyarrow
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import within_ctc_cv as w

ROOT = Path(__file__).resolve().parent.parent
RNG_SEED = w.SEED
DESC = re.compile(r"(\d+(?:\.\d+)?)\s+([A-Za-z]\d+[A-Za-z]*)\s*$")


def parse_desc(s):
    m = DESC.search(str(s)); return (float(m.group(1)), m.group(2)) if m else (np.nan, None)


def build_series(a):
    se = pd.read_parquet(a.series); se = se[~se.is_localizer].copy()
    se["scanner"] = se.manufacturer.str.strip() + " " + se.model.str.strip()
    se["age"] = se.patient_age; se["sex"] = se.patient_sex
    st = pd.read_parquet(a.stage2).copy()
    st["scanner"] = st.manufacturer.str.strip() + " " + st.model.str.strip(); st["age"] = st.age_years
    st["position"] = "supine"
    for d in (se, st):
        pk = d.series_description.map(parse_desc); d["thickness"] = [p[0] for p in pk]; d["kernel"] = [p[1] for p in pk]
    st["is_localizer"] = False
    if Path(a.slices).exists():
        used = set(pd.read_csv(a.slices).series_uid); se = se[se.series_uid.isin(used)]; st = st[st.series_uid.isin(used)]
        print(f"using the {len(used)} series listed in {a.slices}")
    else:
        print(f"{a.slices} not found: using every non-localizer volume series")
    cols = ["patient_id", "series_uid", "scanner", "kernel", "thickness", "image_count", "age", "sex", "position"]
    return se[cols].assign(source="CTC"), st[cols + ["study_date"]].assign(source="StageII")


def per_patient(df):
    mode = lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan
    return df.groupby("patient_id").agg(scanner=("scanner", mode), kernel=("kernel", mode), thickness=("thickness", "median"),
                                        image_count=("image_count", "median"), age=("age", "median"), sex=("sex", mode),
                                        n_series=("series_uid", "size")).reset_index()


def chi2_perm(x, y, rng, n=5000):
    t = pd.crosstab(x, y);
    def stat(a, b):
        tab = pd.crosstab(a, b).values.astype(float); e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        return float(((tab - e) ** 2 / np.where(e > 0, e, 1)).sum())
    obs = stat(x, y); yy = y.copy(); null = []
    for _ in range(n): null.append(stat(x, rng.permutation(yy)))
    return obs, float((1 + (np.array(null) >= obs).sum()) / (1 + n)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=str(ROOT / "data/processed/series.parquet")); ap.add_argument("--stage2", default=str(ROOT / "data/processed/stage2.parquet"))
    ap.add_argument("--polyps", default=str(ROOT / "data/processed/polyps.parquet")); ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--tables", default=str(ROOT / "reports/tables")); ap.add_argument("--fig-dir", default=str(ROOT / "reports/figures"))
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--perm", type=int, default=500)
    a = ap.parse_args(); rng = np.random.default_rng(RNG_SEED); T = Path(a.tables); T.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    cs = pd.read_csv(a.split); cs = cs[cs.split.isin(["train", "val", "test", "challenge"])]   # frozen cohort only (no excluded patients)
    ctc, st = build_series(a)
    P = pd.concat([per_patient(ctc), per_patient(st).assign(sex=lambda d: d.sex)], ignore_index=True)
    P = P.merge(cs[["patient_id", "source", "label", "split"]], on="patient_id", how="inner")

    # ---- Part 1 ----
    print("\n=== Part 1: StageII vs CT Colonography (patients in the frozen cohort) ===")
    rows = []
    for src, g in P.groupby("source"):
        rows.append(dict(source=src, patients=len(g), scanner_top=g.scanner.value_counts().head(3).to_dict(),
                         kernel_top=g.kernel.value_counts().head(3).to_dict(),
                         thickness_median=g.thickness.median(), image_count_median=g.image_count.median(),
                         image_count_range=f"{g.image_count.min():.0f}-{g.image_count.max():.0f}", age_median=g.age.median(),
                         female_share=(lambda s: (s == "F").sum() / max(s.isin(["F", "M"]).sum(), 1))(g.sex.astype(str).str.upper().str[0]), sex_known=int(g.sex.astype(str).str.upper().str[0].isin(["F", "M"]).sum())))
    S = pd.DataFrame(rows); S.to_csv(T / "acq_source_summary.csv", index=False); print(S.T.to_string(header=False))
    y = (P.source == "StageII").astype(int).values
    for c in ["thickness", "image_count", "age"]:
        x = P[c].values; ok = ~np.isnan(x)
        if ok.sum() > 10 and 0 < y[ok].sum() < ok.sum(): print(f"  {c:12s} AUC for telling the source apart: {max(roc_auc_score(y[ok], x[ok]), 1 - roc_auc_score(y[ok], x[ok])):.3f}  (1.0 = perfectly different, 0.5 = no difference)")
    sd = st.study_date.dropna()
    if len(sd): print(f"  StageII scan years: {sd.dt.year.min()}-{sd.dt.year.max()}  (CTC dates are not public)")

    # ---- Part 2 ----
    print("\n=== Part 2: inside CT Colonography ===")
    grp = w.patient_table(a.polyps, a.split)[["patient_id", "group", "split"]]
    C = P[P.source == "CTC"].merge(grp[["patient_id", "group"]], on="patient_id")
    C["id_style"] = np.where(C.patient_id.str.startswith("CTC-"), "CTC-style id", "1.3.6-style id")    # probable collection site / batch
    C.to_csv(T / "acq_within_ctc_features.csv", index=False)
    print(f"{len(C)} labelled CTC patients; scanner models: {C.scanner.value_counts().to_dict()}; kernels: {C.kernel.value_counts().head(5).to_dict()}")
    assoc, model_rows, oofs = [], [], {}
    for task, (pos, neg) in w.TASKS.items():
        D = C[C.group.isin(pos + neg)].copy(); yy = D.group.isin(pos).astype(int).values
        for c in ["scanner", "kernel", "sex", "id_style"]:
            xs = D[c].astype(str)
            if xs.nunique() > 1:
                stat, p, _ = chi2_perm(xs.values, yy, rng); assoc.append(dict(task=task, feature=c, test="permutation chi-square", statistic=stat, p=p))
        for c in ["thickness", "image_count", "age", "n_series"]:
            x = D[c].values; ok = ~np.isnan(x)
            if ok.sum() > 10 and len(np.unique(yy[ok])) == 2 and np.ptp(x[ok]) > 0:
                u = mannwhitneyu(x[ok][yy[ok] == 1], x[ok][yy[ok] == 0]); assoc.append(dict(task=task, feature=c, test="Mann-Whitney", statistic=u.statistic, p=u.pvalue))
        # acquisition-only model
        X = pd.get_dummies(D[["scanner", "kernel", "sex", "id_style"]].astype(str), dtype=float)
        for c in ["thickness", "image_count", "age"]:
            v = D[c].astype(float); X[c] = v.fillna(v.median()).fillna(0)
        X = X.loc[:, X.nunique() > 1].values
        if X.shape[1] == 0 or len(np.unique(yy)) < 2:
            model_rows.append(dict(task=task, note="no variation in acquisition features")); continue
        oof = w.cv_scores(X, yy, 1.0, a.reps, 5); s = oof.mean(0); per = [roc_auc_score(yy, o) for o in oof]
        r = w.summarise(yy, s, per, rng); r.update(task=task, perm_p=w.perm_p(X, yy, 1.0, 5, a.perm, rng, s_fixed=None, reps=3), n_features=X.shape[1])
        model_rows.append(r); oofs[task] = (yy, s)
    A = pd.DataFrame(assoc); A.to_csv(T / "acq_within_ctc_assoc.csv", index=False)
    M = pd.DataFrame(model_rows); M.to_csv(T / "acq_within_ctc_model.csv", index=False)
    print("\nassociation of each feature with the labels (small p = linked):"); print(A.round(4).to_string(index=False))
    print("\nmodel that sees ONLY acquisition details (AUC 0.5 = no information):")
    if "auc" in M: print(M[["task", "n_pos", "n_neg", "auc", "auc_lo", "auc_hi", "perm_p"]].round(3).to_string(index=False))
    bal = pd.crosstab(C.scanner, C.split)
    bal.to_csv(T / "acq_split_balance.csv"); print("\nscanner model by frozen split (patients):"); print(bal.to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4)); ct = pd.crosstab(C.scanner, C.group)
        ct.plot.bar(stacked=True, ax=axs[0], legend=True, fontsize=7); axs[0].set_title("scanner model by label group (CTC)", fontsize=9); axs[0].legend(fontsize=6)
        if oofs:
            for k, (yy, s) in oofs.items():
                from sklearn.metrics import roc_curve; f, t, _ = roc_curve(yy, s); axs[1].plot(f, t, label=f"{k} AUC {roc_auc_score(yy, s):.2f}")
            axs[1].plot([0, 1], [0, 1], "k--", lw=.6); axs[1].set_title("acquisition-only model", fontsize=9); axs[1].legend(fontsize=6)
        fig.tight_layout(); fd = Path(a.fig_dir); fd.mkdir(parents=True, exist_ok=True); fig.savefig(fd / "acq_within_ctc.png", dpi=130); plt.close(fig)
    except Exception as e: print("figure not saved:", e, "(try --fig-dir with another folder)")


if __name__ == "__main__":
    main()

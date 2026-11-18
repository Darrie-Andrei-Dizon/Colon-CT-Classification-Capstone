"""Scan-matched subset check: if StageII and CT Colonography scans are matched on scanner settings, can the source
still be told apart?

    Part 1 only (headers, no images, a few seconds):
        python src/acquisition_match.py
    Part 1 + Part 2 (needs the prepared scans on the D: drive, a few minutes):
        python src/acquisition_match.py --npy-dir D:\\tcia\\processed

Part 1  Overlap of DICOM header settings between the two sources, for the scans the models actually use
        (colon_slices.csv): manufacturer, kernel, slice thickness, kVp, patient position, contrast tag, and the series
        description (StageII 'Venous Phase' = IV contrast given).  Then the matched subset:
            SIEMENS + kernel B30f + 120 kVp + supine (FFS or HFS)
        Slice thickness is NOT matched by default: no CT Colonography scan in use is 5 mm thick, and the pipeline
        resamples every scan to 5 mm slabs anyway.  --strict adds thickness = 5.0 mm and shows how few remain.
Part 2  The colon brightness source check (same features and HU window as source_check_colon.py) on the matched
        patients only.  Because the subset is small, it uses repeated patient-level cross-validation (5 folds x 20
        repeats) with a permutation p-value, not the frozen train/test split.  Also a header-only model on the
        settings that are still free to differ (tube current, pixel size, scan length).
How to read Part 2
        AUC still near 1.0 = matching the scanner settings does not remove the source difference.  What is left is
        the scan type itself (colon inflated with gas for colonography; routine contrast CT with a collapsed colon
        for StageII), which no pre-processing can undo.  Report this in Chapter 4 and the limitations.
        AUC near 0.5 = the settings explained the difference; then the matched subset is worth training on.
The 3 CT Colonography cancers are counted in Part 1 but left out of Part 2 (it compares sources, as the original check).

Outputs: reports/tables/acq_match_overlap.csv     setting x source counts, and whether the setting occurs in both
         reports/tables/acq_match_subset.csv      matched series (patient, source, label, split, settings)
         reports/tables/acq_match_summary.csv     counts before/after matching, contrast evidence
         reports/tables/acq_match_source_check.csv  Part 2 AUCs (only with --npy-dir)
Needs: numpy pandas scipy scikit-learn pyarrow
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_check_colon as sc          # same body mask, HU window and brightness features

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261023
FIELDS = ["manufacturer", "kernel", "slice_thickness", "kvp", "patient_position", "contrast_tag", "contrast_in_desc"]
SUPINE = {"FFS", "HFS"}


def load(a):
    d = pd.read_parquet(a.headers); d = d[d.found].copy()
    used = set(pd.read_csv(a.slices).series_uid); d = d[d.series_uid.isin(used)].copy()
    d["contrast_tag"] = d.contrast.fillna("none recorded").astype(str)
    d["contrast_in_desc"] = d.desc.fillna("").str.contains("venous|contrast|arterial|portal", case=False).map({True: "yes", False: "no"})
    return d


def overlap_table(d):
    rows = []
    for f in FIELDS:
        ct = pd.crosstab(d[f].fillna("missing").astype(str), d.source)
        for val, r in ct.iterrows():
            n_ctc, n_st = int(r.get("CTC", 0)), int(r.get("StageII", 0))
            rows.append(dict(field=f, value=val, ctc_series=n_ctc, stageII_series=n_st, in_both=n_ctc > 0 and n_st > 0))
    return pd.DataFrame(rows)


def matched(d, strict):
    m = (d.manufacturer.str.upper() == "SIEMENS") & (d.kernel == "B30f") & (d.kvp == 120) & d.patient_position.isin(SUPINE)
    if strict: m &= d.slice_thickness == 5.0
    return d[m].copy()


def counts(d, tag):
    out = dict(step=tag)
    for s in ["CTC", "StageII"]:
        x = d[d.source == s]
        out[f"{s}_series"] = len(x); out[f"{s}_patients"] = x.patient_id.nunique()
        out[f"{s}_malignant_patients"] = x[x.label == "malignant"].patient_id.nunique()
    out["StageII_contrast_in_desc"] = int(((d.source == "StageII") & (d.contrast_in_desc == "yes")).sum())
    return out


def colon_features(m, a):
    sl = pd.read_csv(a.slices); sl = sl[sl.series_uid.isin(set(m.series_uid))]
    rows, mdir = [], Path(a.masks)
    for k, (uid, g) in enumerate(sl.groupby("series_uid"), 1):
        z = np.sort(g.slab_idx.values)
        try:
            vol = sc.load_slabs(Path(a.npy_dir) / f"{uid}.npy", z)
            body = sc.body_mask(vol); colon = np.load(mdir / f"{uid}.npz")["mask"][z]
            for region, msk in (("range", body), ("mask", colon & body)):
                if msk.sum() < 500: continue
                rows.append(dict(series_uid=uid, patient_id=g.patient_id.iloc[0], source=g.source.iloc[0], region=region, **sc.features(vol[msk])))
        except Exception as e:
            print("FAILED", uid[-10:], e)
        if k % 25 == 0: print(f"{k}/{sl.series_uid.nunique()} scans", flush=True)
    return pd.DataFrame(rows)


def cv_auc(X, y, groups, reps, rng=None, perm=False):
    if perm: y = rng.permutation(y)
    aucs = []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, groups):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)).fit(X[tr], y[tr])
            s[te] = mdl.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs)), aucs


def probe(name, region, X, y, groups, n_perm):
    rng = np.random.default_rng(SEED)
    auc, per = cv_auc(X, y, groups, 20)
    null = [cv_auc(X, y, groups, 1, rng, perm=True)[0] for _ in range(n_perm)]
    p = (1 + sum(n >= auc for n in null)) / (1 + n_perm)
    return dict(features=name, region=region, auc=auc, auc_repeat_lo=np.percentile(per, 2.5), auc_repeat_hi=np.percentile(per, 97.5),
                perm_p=p, n_ctc_patients=len(set(groups[y == 0])), n_stageII_patients=len(set(groups[y == 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers", default=str(ROOT / "data/processed/dicom_series.parquet"))
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--masks", default=str(ROOT / "data/processed/colon"))
    ap.add_argument("--npy-dir", help="prepared scans (e.g. D:\\tcia\\processed); without it only Part 1 runs")
    ap.add_argument("--strict", action="store_true", help="also require 5.0 mm slice thickness")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out-dir", default=str(ROOT / "reports/tables"))
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    # Part 1
    d = load(a); ov = overlap_table(d); ov.to_csv(out / "acq_match_overlap.csv", index=False)
    m = matched(d, a.strict)
    keep = ["patient_id", "source", "label", "split", "series_uid", "desc", "manufacturer", "kernel", "kvp", "patient_position",
            "slice_thickness", "pixel_mm", "tube_current", "z_extent_mm", "contrast_tag", "contrast_in_desc"]
    m[keep].to_csv(out / "acq_match_subset.csv", index=False)
    summ = pd.DataFrame([counts(d, "all scans in use"), counts(m, "matched" + (" (strict, 5 mm)" if a.strict else ""))])
    summ.to_csv(out / "acq_match_summary.csv", index=False)
    print("\nPart 1: settings found in only one source (in_both = False)")
    print(ov[~ov.in_both].to_string(index=False))
    print("\nPart 1: counts before and after matching")
    print(summ.T.to_string(header=False))
    if not a.npy_dir:
        print("\nPart 2 skipped (no --npy-dir)."); return

    # Part 2
    if not Path(a.npy_dir).exists(): raise SystemExit(f"cannot reach {a.npy_dir} (is the D: drive connected?)")
    m2 = m[m.label.eq("benign") | m.source.eq("StageII")]          # leave out the CT Colonography cancers
    if m2.source.nunique() < 2 or (m2.source == "CTC").sum() < 10: raise SystemExit("too few matched scans in one source for Part 2")
    F = colon_features(m2, a)
    res = []
    for region in ["range", "mask"]:
        f = F[F.region == region]
        if f.source.nunique() < 2: continue
        y, g = (f.source == "StageII").astype(int).values, f.patient_id.values
        for name, cols in {"4 summaries": ["p1", "p50", "p99", "dark_frac"], "32-bin histogram": [f"h{i:02d}" for i in range(32)]}.items():
            res.append(probe(name, region, f[cols].values, y, g, a.n_perm))
    h = m2.dropna(subset=["tube_current", "pixel_mm", "z_extent_mm"])
    res.append(probe("header only: tube current, pixel size, scan length", "header", h[["tube_current", "pixel_mm", "z_extent_mm"]].values,
                     (h.source == "StageII").astype(int).values, h.patient_id.values, a.n_perm))
    R = pd.DataFrame(res); R.to_csv(out / "acq_match_source_check.csv", index=False)
    print("\nPart 2: can the source still be told apart in the matched subset? (repeated 5-fold CV)")
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

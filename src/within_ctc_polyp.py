"""Within-CT-Colonography analysis on the slabs around each recorded polyp (Wesp-style, 2D, no in-plane crop).

Why: within_ctc_cv.py averaged backbone features over the WHOLE colon, so a 5-10 mm polyp is a tiny part of what the
model sees.  Wesp et al. (Eur Radiol 2022) classified polyps from small subvolumes around each polyp.  We cannot crop
in-plane (no polyp coordinates), but the ACRIN sheets give supine / prone SLICE NUMBERS for part of the cohort, so we
keep only the slabs within +/- K slabs (5 mm each) of those slices.

Three steps (run from the project folder, on the laptop with the D: drive and the GPU):

  1) map slice numbers to slabs and write the slab list (reads DICOM headers only, no pixels; a few minutes)
     python src/within_ctc_polyp.py map --k 1
       -> data/processed/colon_slices_polyp_k1.csv   (same columns as colon_slices.csv)
       -> reports/tables/within_ctc_polyp_mapping.csv (one row per listed slice: series, slab, inside colon range?)
  2) features from the polyp slabs (existing script, unchanged), ImageNet and DenseCL ResNet-50:
     python src/extract_embeddings.py --arch resnet50 --npy-dir D:\\tcia\\processed --slices data\\processed\\colon_slices_polyp_k1.csv --out D:\\runs\\emb_imagenet_resnet_polyp_k1.parquet
     python src/extract_embeddings.py --arch resnet50 --encoder D:\\runs\\dcl_resnet_raw\\encoder.pt --npy-dir D:\\tcia\\processed --slices data\\processed\\colon_slices_polyp_k1.csv --out D:\\runs\\emb_dcl_resnet_polyp_k1.parquet
  3) compare on the SAME patients (only those with a mapped slice): size rule, whole-colon features, polyp-slab features
     python src/within_ctc_polyp.py compare --embeddings imagenet_whole=D:\\runs\\emb_imagenet_resnet.parquet imagenet_polyp=D:\\runs\\emb_imagenet_resnet_polyp_k1.parquet dcl_whole=D:\\runs\\emb_dcl_resnet.parquet dcl_polyp=D:\\runs\\emb_dcl_resnet_polyp_k1.parquet
       -> reports/tables/within_ctc_polyp_compare.csv
     (whole-colon parquet files: the ones made earlier for within_ctc_cv.py; if lost, re-run extract_embeddings.py without --slices)

Slice-number reading.  The sheets do not say whether a slice number is the DICOM InstanceNumber or a position counted
from the top of the series.  Both readings are mapped; the script keeps the one whose slabs fall inside the automatic colon
range (colon_series.csv) more often.  That rule uses no labels, so it cannot favour a result.  Slice numbers are given per
patient and position, not per polyp, so every listed slice of a patient is used.
Pre-specified: K = 1 (3 slabs, 15 mm) is the main window; K = 2 (25 mm) is a sensitivity check.  Exploratory analysis.
Needs: pydicom numpy pandas pyarrow scikit-learn scipy (all already used by the project)
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
SLAB_MM = 5.0
PRONE, SUPINE = {"HFP", "FFP"}, {"HFS", "FFS"}


def slice_numbers(text):
    """'353/432' -> [353, 432];  '144 Pos >=10mm' -> [144];  empty -> []"""
    if text is None or (isinstance(text, float) and np.isnan(text)): return []
    t = re.sub(r">=?\s*\d+\s*mm", " ", str(text), flags=re.I)
    return [int(v) for v in re.findall(r"\d+", t) if int(v) >= 1]


def header_geometry(folder):
    """Per file: InstanceNumber and z position, plus position (prone/supine) - same z and slab rule as preprocess.py."""
    import pydicom
    files = sorted(Path(folder).glob("*.dcm")); rows = []; pp = None; n = None
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", "PatientPosition"])
        if n is None:
            iop = np.array(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float); n = np.cross(iop[:3], iop[3:])
            pp = str(getattr(ds, "PatientPosition", "")).upper()
        rows.append((int(getattr(ds, "InstanceNumber", 0) or 0), float(np.dot(n, np.array(ds.ImagePositionPatient, float)))))
    g = pd.DataFrame(rows, columns=["instance", "z"])
    zs = np.sort(g.z.unique()); keep = np.r_[True, np.diff(zs) > 1e-3]; zs = zs[keep]          # preprocess: sorted, duplicates dropped
    b = np.round((zs - zs[0]) / SLAB_MM).astype(int); ub = np.unique(b)
    slab_of_z = dict(zip(zs, np.searchsorted(ub, b)))                                       # slab index = rank of the 5 mm bin
    def slab(z): return slab_of_z[zs[np.argmin(np.abs(zs - z))]]
    g["slab"] = g.z.map(slab)
    position = "prone" if pp in PRONE else "supine" if pp in SUPINE else "unknown"
    # reading A: slice number = InstanceNumber;  reading B: slice number = position counted from the top (head end)
    by_instance = dict(zip(g.instance, g.slab))
    from_top = [slab(z) for z in zs[::-1]]
    return position, by_instance, from_top, len(ub)


def cmd_map(a):
    pol = pd.read_parquet(ROOT / "data/processed/polyps.parquet")
    cs = pd.read_csv(ROOT / "data/processed/cohort_split.csv")
    ctc = cs[(cs.source == "CTC") & cs.split.isin(["train", "val", "test"]) & cs.label.isin(["benign"])]
    ser = pd.read_csv(ROOT / "data/processed/colon_series.csv"); ser = ser[ser.patient_id.isin(ctc.patient_id) & (ser.status == "ok")]
    dic = pd.read_parquet(ROOT / "data/processed/dicom_series.parquet")[["series_uid", "folder"]]
    ser = ser.merge(dic, on="series_uid", how="left")
    sl = pol.groupby("patient_id")[["supine_slices_patient", "prone_slices_patient"]].first()
    probe = ser.folder.dropna()
    if probe.empty or not any(Path(f).is_dir() for f in probe.head(5)):
        raise SystemExit(f"Cannot reach the DICOM folders (first one: {probe.iloc[0] if len(probe) else 'none'}).\n"
                         "Is the drive that holds them plugged in and showing in File Explorer?  Then run this again.")
    rows = []
    for k, r in enumerate(ser.itertuples(), 1):
        if r.patient_id not in sl.index: continue
        want = {"supine": slice_numbers(sl.loc[r.patient_id, "supine_slices_patient"]), "prone": slice_numbers(sl.loc[r.patient_id, "prone_slices_patient"])}
        if not (want["supine"] or want["prone"]): continue
        try: pos, by_inst, from_top, nsl = header_geometry(r.folder)
        except Exception as e: print("header read failed", r.series_uid[-10:], e); continue
        if nsl != r.n_slabs: print(f"WARNING {r.series_uid[-10:]}: {nsl} slabs from headers vs {r.n_slabs} in colon_series.csv")
        for p in (["supine", "prone"] if pos == "unknown" else [pos]):
            for s in want[p]:
                for reading, sb in [("instance", by_inst.get(s)), ("from_top", from_top[s - 1] if 1 <= s <= len(from_top) else None)]:
                    rows.append(dict(patient_id=r.patient_id, series_uid=r.series_uid, position=p, header_position=pos, slice_no=s, reading=reading,
                                     slab_idx=sb, n_slabs=r.n_slabs, first_slab=r.first_slab, last_slab=r.last_slab,
                                     in_colon=(sb is not None) and (r.first_slab <= sb <= r.last_slab)))
        if k % 20 == 0: print(f"{k}/{len(ser)} series", flush=True)
    m = pd.DataFrame(rows)
    if m.empty: raise SystemExit("no slice numbers could be mapped - check the polyps.parquet slice columns and the DICOM folders")
    cov = m.groupby("reading").agg(mapped=("slab_idx", lambda v: v.notna().mean()), in_colon=("in_colon", "mean"))
    print("\nlabel-free check of the two readings (share of listed slices that land inside the automatic colon range):\n", cov.round(3))
    best = cov.in_colon.idxmax(); print(f"-> using reading '{best}'")
    m["chosen"] = m.reading == best
    Path(ROOT / "reports/tables").mkdir(parents=True, exist_ok=True); m.to_csv(ROOT / "reports/tables/within_ctc_polyp_mapping.csv", index=False)
    use = m[m.chosen & m.slab_idx.notna()]
    out = []
    for r in use.itertuples():
        for d in range(-a.k, a.k + 1):
            j = int(r.slab_idx) + d
            if 0 <= j < r.n_slabs: out.append(dict(patient_id=r.patient_id, series_uid=r.series_uid, slab_idx=j))
    o = pd.DataFrame(out).drop_duplicates()
    base = pd.read_csv(ROOT / "data/processed/colon_slices.csv").drop_duplicates("series_uid")[["series_uid", "source", "label", "split"]]
    o = o.merge(base, on="series_uid", how="left"); o["colon_area_px"] = np.nan
    o = o.merge(ser[["series_uid", "n_slabs"]], on="series_uid")
    o = o[["patient_id", "source", "label", "split", "series_uid", "slab_idx", "colon_area_px", "n_slabs"]].sort_values(["series_uid", "slab_idx"])
    f = ROOT / f"data/processed/colon_slices_polyp_k{a.k}.csv"; o.to_csv(f, index=False)
    print(f"wrote {f}: {o.patient_id.nunique()} patients, {o.series_uid.nunique()} scans, {len(o)} slabs (K = {a.k})")


def cmd_compare(a):
    from sklearn.metrics import roc_auc_score
    import within_ctc_cv as w
    pt = w.patient_table(ROOT / "data/processed/polyps.parquet", ROOT / "data/processed/cohort_split.csv")
    keep = set(pd.read_csv(ROOT / f"data/processed/colon_slices_polyp_k{a.k}.csv").patient_id)
    pt = pt[pt.patient_id.isin(keep)].reset_index(drop=True)
    print("patients with polyp slabs, per group:", pt.group.value_counts().to_dict())
    feats = {}
    for spec in a.embeddings:
        name, path = spec.split("=", 1); f = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        feats[name] = f
    common = set(pt.patient_id)
    for f in feats.values(): common &= set(f.patient_id)
    pt = pt[pt.patient_id.isin(common)].reset_index(drop=True); print(f"common patients across all inputs: {len(pt)}")
    rng = np.random.default_rng(w.SEED); out = []
    for task, (posg, negg) in w.TASKS.items():
        t = pt[pt.group.isin(posg + negg)].reset_index(drop=True); y = t.group.isin(posg).astype(int).values
        if min(y.sum(), (1 - y).sum()) < a.folds: print(f"skip {task}: too few patients ({y.sum()} vs {(1 - y).sum()})"); continue
        s = np.clip(t.max_polyp_mm.values / 20.0, 0, 1); r = w.summarise(y, s, [], rng)
        r.update(task=task, model="size_rule", p_perm=w.perm_p(None, y, None, a.folds, a.perm, rng, s_fixed=s)); out.append(r)
        for name, f in feats.items():
            d = t[["patient_id"]].merge(f, on="patient_id", how="left"); cols = [c for c in f.columns if c != "patient_id"]
            X = d[cols].values.astype(float); oof = w.cv_scores(X, y, a.C, a.repeats, a.folds)
            rep = [roc_auc_score(y, oof[i]) for i in range(a.repeats)]; r = w.summarise(y, oof.mean(0), rep, rng)
            r.update(task=task, model=name, p_perm=w.perm_p(X, y, a.C, a.folds, a.perm, rng) if a.perm else np.nan); out.append(r)
    res = pd.DataFrame(out); f = ROOT / f"reports/tables/within_ctc_polyp_compare_k{a.k}.csv"; res.to_csv(f, index=False)
    pd.set_option("display.width", 220)
    print(res[["task", "model", "n_pos", "n_neg", "auc", "auc_lo", "auc_hi", "auc_repeat_sd", "p_perm"]].round(3).to_string(index=False))
    print(f"\nwrote {f}.  Reference: Wesp et al. 2022 AUC 0.75 (no masks) to 0.83 (masks), adenoma vs hyperplastic/normal, polyp-centred 3D.")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map"); m.add_argument("--k", type=int, default=1)
    c = sub.add_parser("compare"); c.add_argument("--k", type=int, default=1); c.add_argument("--embeddings", nargs="+", required=True)
    c.add_argument("--C", type=float, default=0.05); c.add_argument("--folds", type=int, default=5); c.add_argument("--repeats", type=int, default=20)
    c.add_argument("--perm", type=int, default=200)
    a = ap.parse_args(); cmd_map(a) if a.cmd == "map" else cmd_compare(a)


if __name__ == "__main__":
    main()

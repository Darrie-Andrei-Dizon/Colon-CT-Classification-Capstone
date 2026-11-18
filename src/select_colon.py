"""Select the colon region of every prepared scan with an automated segmenter (TotalSegmentator).

    python src/select_colon.py --device gpu                 # all prepared scans (resumable)
    python src/select_colon.py --limit 6 --device cpu       # try a few first
    python src/select_colon.py --fast                       # 3 mm model, much quicker, a little less exact

Why it runs on the PREPARED scans (D:\\tcia\\processed\\*.npy) and not on the raw DICOM:
the mask then has exactly the same slab indices as the images the models will see, so there is no
millimetre-to-slab conversion to get wrong. The prepared array is (slab, row, col) in an anatomical frame
(col -> patient left, row -> posterior, slab -> head), 1.5 mm in-plane, 5 mm per slab; the NIfTI wrapper below
states that, so the segmenter sees a correctly oriented scan.

Per series:  segment colon -> keep the main colon pieces -> first/last slab with enough colon -> add a margin.

Outputs
  data/processed/colon_slices.csv    one row per kept slab: patient, series, slab, label, split, colon area
  data/processed/colon_series.csv    one row per series: range, status, QC flags
  data/processed/colon/<series_uid>.npz   the 3D colon mask (bool), for later use (e.g. masking, DenseCL crops)
  reports/figures/colon_qc.png       coronal + axial check images: all flagged series plus a random sample

The segmenter is pretrained and never sees our labels, so running it on validation and test scans is not leakage.
Needs: pip install totalsegmentator nibabel scipy pandas pyarrow matplotlib   (first run downloads the model weights).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
SLAB_MM, PIX_MM = 5.0, 1.5
SEED = 20261023                                   # only used to choose which series appear in the QC sample


def totalseg_backend(device="gpu", fast=False):
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    # array (slab,row,col) = (S, P, L)  ->  NIfTI (x,y,z) with RAS+ affine: col->left (-x), row->posterior (-y), slab->superior (+z)
    affine = np.diag([-PIX_MM, -PIX_MM, SLAB_MM, 1.0])

    def segment(vol):
        img = nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32), affine)
        out = totalsegmentator(img, None, ml=True, task="total", roi_subset=["colon"], fast=fast,
                               device=device, skip_saving=True, quiet=True)
        m = np.asarray(out.get_fdata()) > 0                   # only 'colon' was requested
        return np.ascontiguousarray(m.transpose(2, 1, 0))
    return segment


def mock_backend():
    """TEST ONLY: air-filled voxels inside the body in the lower half of the image. Not a real colon."""
    def segment(vol):
        body = np.stack([ndi.binary_fill_holes(s > -300) for s in vol])
        m = body & (vol < -800)
        m[:, : vol.shape[1] // 2, :] = False
        return ndi.binary_opening(m, iterations=1)
    return segment


def pick_range(mask, min_area, margin, keep_frac=0.10):
    """Colon pieces = 3D components >= keep_frac of the largest; range = first..last slab with >= min_area colon pixels."""
    lab, n = ndi.label(mask)
    if n == 0:
        return None, mask & False
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= keep_frac * sizes.max()))
    area = keep.sum((1, 2))
    ok = np.flatnonzero(area >= min_area)
    if len(ok) == 0:
        return None, keep
    return (max(int(ok[0]) - margin, 0), min(int(ok[-1]) + margin, mask.shape[0] - 1)), keep


def qc_figure(items, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols * 2, figsize=(cols * 5.2, rows * 3.4), squeeze=False)
    for ax in axs.ravel(): ax.axis("off")
    for k, (title, vol, mask, rng) in enumerate(items):
        r, c = divmod(k, cols); a_cor, a_ax = axs[r, 2 * c], axs[r, 2 * c + 1]
        ys = np.flatnonzero(mask.any((0, 2))); y = int(np.median(ys)) if len(ys) else vol.shape[1] // 2
        a_cor.imshow(vol[:, y, :], cmap="gray", vmin=-400, vmax=400, origin="lower", aspect=SLAB_MM / PIX_MM)
        if mask[:, y, :].any(): a_cor.contour(mask[:, y, :].astype(float), [0.5], colors="#e4572e", linewidths=0.8)
        if rng: [a_cor.axhline(v, color="#2e86e4", lw=0.8) for v in rng]
        a_cor.set_title(title, fontsize=7, loc="left")
        zs = np.flatnonzero(mask.any((1, 2))); z = int(np.median(zs)) if len(zs) else vol.shape[0] // 2
        a_ax.imshow(vol[z], cmap="gray", vmin=-400, vmax=400)
        if mask[z].any(): a_ax.contour(mask[z].astype(float), [0.5], colors="#e4572e", linewidths=0.8)
        a_ax.set_title(f"axial slab {z}", fontsize=7, loc="left")
        a_cor.axis("off"); a_ax.axis("off")
    fig.suptitle("Colon selection check: red = colon mask, blue lines = kept range (left: coronal, right: axial)", fontsize=9)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", default=str(ROOT / "data/processed/preproc_series.parquet"))
    ap.add_argument("--download", default=str(ROOT / "data/manifests/download_series.csv"),
                    help="series -> patient/source/split (the frozen manifest)")
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"), help="patient -> label")
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--fig", default=str(ROOT / "reports/figures/colon_qc.png"))
    ap.add_argument("--backend", choices=["totalseg", "mock"], default="totalseg")
    ap.add_argument("--device", default="gpu", help="gpu | cpu | mps")
    ap.add_argument("--fast", action="store_true", help="TotalSegmentator 3 mm model")
    ap.add_argument("--margin", type=int, default=1, help="extra slabs kept either side of the colon (5 mm each)")
    ap.add_argument("--min-area", type=int, default=150, help="min colon pixels (1.5 mm) in a slab to count it as colon")
    ap.add_argument("--min-slabs", type=int, default=20, help="flag ranges shorter than this (20 slabs = 10 cm)")
    ap.add_argument("--qc", type=int, default=12, help="random series to add to the QC image (flagged ones are always added)")
    ap.add_argument("--qc-source", choices=["StageII", "CTC"], help="show only this dataset in the QC image")
    ap.add_argument("--qc-max", type=int, default=24, help="max series in the QC image")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); mdir = out / "colon"; mdir.mkdir(parents=True, exist_ok=True)
    pre = pd.read_parquet(a.preproc); pre = pre[pre.ok == True][["series_uid", "out_file"]]
    man = pd.read_csv(a.download)[["series_uid", "patient_id", "source", "split"]].drop_duplicates("series_uid")
    lab = pd.read_csv(a.split)[["patient_id", "label"]]
    d = pre.merge(man, on="series_uid", how="left").merge(lab, on="patient_id", how="left")
    missing = d.patient_id.isna().sum()
    if missing: print(f"note: {missing} prepared series are not in the download manifest (kept, with blank patient/split)")
    if a.limit: d = pd.concat([d[d.source == "StageII"].head(a.limit // 2), d[d.source == "CTC"].head(a.limit - a.limit // 2)])
    segment = mock_backend() if a.backend == "mock" else totalseg_backend(a.device, a.fast)
    if a.backend == "mock": print("WARNING: mock backend is for testing the script only - the output is NOT a colon selection")

    ser_rows, slab_rows, cache = [], [], {}
    for i, r in enumerate(d.itertuples(), 1):
        mfile = mdir / f"{r.series_uid}.npz"
        try:
            vol = np.load(r.out_file).astype(np.float32)
            if mfile.exists(): mask = np.load(mfile)["mask"]
            else:
                mask = segment(vol).astype(bool)
                if mask.shape != vol.shape: raise ValueError(f"mask shape {mask.shape} != image shape {vol.shape}")
            rng, mask = pick_range(mask, a.min_area, a.margin)
            np.savez_compressed(mfile, mask=mask)
            n = vol.shape[0]; flags = []
            if rng is None: status = "no_colon"
            else:
                status = "ok"; lo, hi = rng
                if hi - lo + 1 < a.min_slabs: flags.append("short_range")
                if lo == 0 or hi == n - 1: flags.append("touches_scan_edge")
                area = mask.sum((1, 2))
                for z in range(lo, hi + 1):
                    slab_rows.append(dict(patient_id=r.patient_id, source=r.source, label=r.label, split=r.split,
                                          series_uid=r.series_uid, slab_idx=z, colon_area_px=int(area[z]), n_slabs=n))
            ser_rows.append(dict(series_uid=r.series_uid, patient_id=r.patient_id, source=r.source, label=r.label, split=r.split,
                                 n_slabs=n, first_slab=None if rng is None else rng[0], last_slab=None if rng is None else rng[1],
                                 n_kept=0 if rng is None else rng[1] - rng[0] + 1,
                                 colon_voxels=int(mask.sum()), status=status, qc_flags=";".join(flags)))
            cache[r.series_uid] = (rng, bool(flags) or status != "ok")
        except Exception as e:
            ser_rows.append(dict(series_uid=r.series_uid, patient_id=r.patient_id, source=r.source, label=r.label,
                                 split=r.split, status="error", qc_flags=str(e)[:150])); print("FAILED", r.series_uid[-10:], e)
        if i % 10 == 0 or i == len(d): print(f"{i}/{len(d)}", flush=True)

    ss, sl = pd.DataFrame(ser_rows), pd.DataFrame(slab_rows)
    ss.to_csv(out / "colon_series.csv", index=False); sl.to_csv(out / "colon_slices.csv", index=False)
    print(f"\nseries: {len(ss)}  | status: {ss.status.value_counts().to_dict()}  | flagged: {int((ss['qc_flags'].fillna('') != '').sum())}")
    ok = ss[ss.status == "ok"]
    if len(ok): print("kept slabs per series (5 mm each):\n", ok.groupby("source").n_kept.describe().round(1).to_string())

    # QC image: every flagged / failed series plus a seeded random sample of the rest
    qs = ss if not a.qc_source else ss[ss.source == a.qc_source]
    flagged = (qs.status != "ok") | (qs["qc_flags"].fillna("") != "")
    rng_qc = np.random.default_rng(SEED)
    bad = list(rng_qc.permutation(qs[flagged].series_uid.tolist()))     # seeded shuffle: not just the first scans in the list
    rest = qs[~flagged].series_uid.tolist()
    pick = (bad + list(rng_qc.permutation(rest)[: a.qc]))[: a.qc_max]
    items = []
    for u in pick:
        row = d[d.series_uid == u].iloc[0]
        if not (mdir / f"{u}.npz").exists(): continue
        vol = np.load(row.out_file).astype(np.float32); mask = np.load(mdir / f"{u}.npz")["mask"]
        s = ss[ss.series_uid == u].iloc[0]
        rng = None if pd.isna(s.first_slab) else (int(s.first_slab), int(s.last_slab))
        items.append((f"{row.source} {str(row.patient_id)[-8:]} {row.label} {row.split} {s['qc_flags'] or ''}", vol, mask, rng))
    if items: qc_figure(items, Path(a.fig)); print("QC image:", a.fig)


if __name__ == "__main__":
    main()

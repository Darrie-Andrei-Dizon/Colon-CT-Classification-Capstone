"""Check the automatic colon selection (step 7, TotalSegmentator fast model) against expert colon masks (HQColon).

HQColon (Finocchiaro et al., Sci Data 2026): 435 radiologist-validated full-colon masks (gas AND fluid) made on TCIA CT
Colonography scans, same grid as the original CT, .mha.  62 of our kept CT Colonography scans (43 labelled patients) are
among them (data/external/hqcolon_overlap_our_scans.csv).  The zip files are read member by member in memory; nothing is
unpacked to disk (each mask is ~150 MB unpacked).

Level 1 (this script): agreement along the body axis, in OUR 5 mm slabs, because slab selection is what step 7 decides.
  expert slabs  E = slabs with >= MIN_EXPERT_VOX expert colon voxels (mask slices binned exactly as preprocess.py does)
  our range     K = first_slab..last_slab in colon_series.csv (what the models see; includes the 1-slab margin)
  our mask      O = slabs where our colon mask (data/processed/colon/<uid>.npz) has >= MIN_OUR_PX pixels
  their TS      T = same as E but from HQColon's own TotalSegmentator mask (full model, full resolution), for reference
Per scan: coverage = |E in K| / |E|; slabs of E missed below K (rectal end) and above K; extra kept slabs without colon;
slab-level Dice(E, O) and Dice(E, T); lowest-slab offsets O - E and T - E (positive = starts higher, i.e. misses rectum);
correlation of colon area per slab (ours vs expert, mm^2).
Orientation: slab 0 is the feet end in preprocess.py.  The mask's slab order is matched by its world z; if the two ends
look swapped (area profiles anti-correlated) the scan is flagged and the flipped order used - a geometry step, no labels.
Pass rules, fixed BEFORE any result was seen (7 Nov 2026): median coverage >= 0.95, and no scan misses > 2 slabs (10 mm)
at the rectal end.
Run:  python src/check_colon_vs_hqcolon.py          (resumable: finished scans are cached in the per-scan table)
Out:  reports/tables/colon_vs_hqcolon_scans.csv, reports/tables/colon_vs_hqcolon_summary.csv
Needs: numpy pandas (no SimpleITK, no DICOM, no GPU)
"""
import argparse, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SLAB_MM, PIX_MM = 5.0, 1.5
MIN_EXPERT_VOX, MIN_OUR_PX = 100, 20          # ~45 mm2 each; ignores specks
HQ = ROOT / "data/external/hqcolon"


def read_mha(zf, name):
    raw = zf.read(name); end = raw.index(b"ElementDataFile"); end = raw.index(b"\n", end) + 1
    h = {}
    for line in raw[:end].decode().splitlines():
        if "=" in line: k, v = line.split("=", 1); h[k.strip()] = v.strip()
    nx, ny, nz = map(int, h["DimSize"].split()); sx, sy, sz = map(float, h["ElementSpacing"].split())
    if h.get("CompressedData", "False") == "True": raise ValueError("compressed mha not supported")
    if h["ElementType"] not in ("MET_UCHAR", "MET_CHAR"): raise ValueError(h["ElementType"])
    vol = np.frombuffer(raw, np.uint8, nx * ny * nz, end).reshape(nz, ny, nx)
    M = np.array(list(map(float, h["TransformMatrix"].split()))).reshape(3, 3); off = np.array(list(map(float, h["Offset"].split())))
    z = off[2] + np.arange(nz) * sz * M[2, 2]                      # world z (LPS, head = +) of each slice
    return vol, z, sx * sy


def per_slab(vol, z, vox_mm2):
    """voxel counts and colon area (mm2, mean over the slab's slices) per 5 mm slab, preprocess.py binning."""
    order = np.argsort(z); z = z[order]; cnt = (vol[order] > 0).reshape(len(z), -1).sum(1)
    keep = np.r_[True, np.diff(z) > 1e-3]; z, cnt = z[keep], cnt[keep]
    b = np.round((z - z[0]) / SLAB_MM).astype(int); ub, inv = np.unique(b, return_inverse=True)
    vox = np.bincount(inv, weights=cnt, minlength=len(ub)); n = np.bincount(inv, minlength=len(ub))
    return vox, vox / n * vox_mm2


def dice(a, b): return 2 * len(a & b) / (len(a) + len(b)) if (a or b) else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (resume later)")
    a = ap.parse_args(); t0 = time.time()
    meta = pd.read_json(HQ / "meta-data.json", lines=True)
    ov = pd.read_csv(ROOT / "data/external/hqcolon_overlap_our_scans.csv")
    ser = pd.read_csv(ROOT / "data/processed/colon_series.csv").set_index("series_uid")
    cs = pd.read_csv(ROOT / "data/processed/cohort_split.csv").set_index("patient_id")
    ov = ov.merge(meta[["InstanceUID", "nnunet_label_file"]], left_on="series_uid", right_on="InstanceUID")
    out_f = ROOT / "reports/tables/colon_vs_hqcolon_scans.csv"; out_f.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_f) if out_f.exists() else pd.DataFrame(columns=["series_uid"]); rows = done.to_dict("records")
    zfull = zipfile.ZipFile(HQ / "gas-and-fluid-filled-colon-segmentation.zip"); zts = zipfile.ZipFile(HQ / "masks-totalsegmentator.zip")
    full_names = {Path(n).name: n for n in zfull.namelist()}; ts_names = {Path(n).name: n for n in zts.namelist()}
    todo = ov[~ov.series_uid.isin(done.series_uid)]
    print(f"{len(ov)} matched scans, {len(done)} already done, {len(todo)} to do", flush=True)
    for k, r in enumerate(todo.itertuples(), 1):
        if a.max_seconds and time.time() - t0 > a.max_seconds: print("time limit reached; run again to continue"); break
        s = ser.loc[r.series_uid]; lab = r.nnunet_label_file
        ev, ez, va = read_mha(zfull, full_names[lab]); e_vox, e_area = per_slab(ev, ez, va); del ev
        tname = ts_names.get(lab.replace(".mha", "_mask.mha"))
        if tname: tv, tz, tva = read_mha(zts, tname); t_vox, _ = per_slab(tv, tz, tva); del tv
        om = np.load(ROOT / f"data/processed/colon/{r.series_uid}.npz")["mask"]; o_px = om.reshape(len(om), -1).sum(1)
        o_area = o_px * PIX_MM ** 2
        n = int(s.n_slabs); flipped = False; note = ""
        if len(e_vox) != n: note = f"slab count {len(e_vox)} (mask) vs {n} (ours)"
        m = min(len(e_vox), n, len(o_area))
        c_fwd = np.corrcoef(e_area[:m], o_area[:m])[0, 1]; c_rev = np.corrcoef(e_area[::-1][:m], o_area[::-1][:m])[0, 1] if len(e_vox) == n else np.nan
        if len(e_vox) == n and c_rev > c_fwd + 0.2:
            flipped = True; e_vox, e_area = e_vox[::-1], e_area[::-1]
            if tname: t_vox = t_vox[::-1]
        E = set(np.flatnonzero(e_vox >= MIN_EXPERT_VOX)); O = set(np.flatnonzero(o_px >= MIN_OUR_PX))
        T = set(np.flatnonzero(t_vox >= MIN_EXPERT_VOX)) if tname else set()
        K = set(range(int(s.first_slab), int(s.last_slab) + 1))
        row = dict(series_uid=r.series_uid, patient_id=r.patient_id, position=r.position, split=cs.loc[r.patient_id, "split"],
                   mask_file=lab, n_slabs=n, flipped=flipped, note=note, expert_slabs=len(E), kept_slabs=len(K),
                   coverage=len(E & K) / len(E) if E else np.nan,
                   missed_low=len([e for e in E if e < s.first_slab]), missed_high=len([e for e in E if e > s.last_slab]),
                   extra_kept=len(K - E), dice_ours=dice(E, O), dice_their_ts=dice(E, T) if tname else np.nan,
                   low_offset_ours=(min(O) - min(E)) if O and E else np.nan, low_offset_their_ts=(min(T) - min(E)) if T and E else np.nan,
                   high_offset_ours=(max(O) - max(E)) if O and E else np.nan,
                   area_corr=float(np.corrcoef(e_area[:m], o_area[:m])[0, 1]))
        rows.append(row); pd.DataFrame(rows).to_csv(out_f, index=False)
        print(f"{len(rows)}/{len(ov)} {lab}: coverage {row['coverage']:.2f}, missed low {row['missed_low']}, dice {row['dice_ours']:.2f}{' FLIPPED' if flipped else ''} {note}", flush=True)
    d = pd.DataFrame(rows)
    if len(d) == len(ov):
        summ = dict(scans=len(d), patients=d.patient_id.nunique(), flipped=int(d.flipped.sum()), slab_count_mismatch=int((d.note.fillna("") != "").sum()),
                    coverage_median=d.coverage.median(), coverage_min=d.coverage.min(),
                    scans_missing_gt2_low=int((d.missed_low > 2).sum()), missed_low_max=int(d.missed_low.max()),
                    dice_ours_median=d.dice_ours.median(), dice_their_ts_median=d.dice_their_ts.median(),
                    low_offset_ours_median=d.low_offset_ours.median(), low_offset_their_ts_median=d.low_offset_their_ts.median(),
                    area_corr_median=d.area_corr.median())
        summ["PASS"] = bool(summ["coverage_median"] >= 0.95 and summ["scans_missing_gt2_low"] == 0)
        pd.Series(summ).to_csv(ROOT / "reports/tables/colon_vs_hqcolon_summary.csv"); print(pd.Series(summ).to_string())


if __name__ == "__main__":
    main()

"""Freeze the cohort, the patient-level split, and the download manifests.

Run AFTER build_polyp_table.py and BEFORE any download-dependent modelling:
    python src/make_split.py

Outputs (data/processed/ and data/manifests/):
    cohort_split.csv       one row per patient: source, label, split
    split_meta.json        seed, fractions, counts, SHA-256 of cohort_split.csv
    download_series.csv    every series to download (+ GB, split)
    stageII.tcia           NBIA Data Retriever manifest, StageII cancers
    ctc_core.tcia          NBIA Data Retriever manifest, CTC benign + CTC cancers (volumes only)

Design (matches the approved methodology):
  * Split unit is the PATIENT, so prone and supine scans of one patient never straddle splits.
  * Malignant = StageII (230).  Benign = CTC patients with >=1 non-malignant polyp and no malignant one.
  * The 3 CTC cancers are a held-out CHALLENGE set, never used for training or tuning.
  * Excluded: 2 patients in conflicting label files; patients whose only polyps are histology
    'other / not applicable'; CTC patients with no polyp (out of scope, section 1.5.1).
  * CTC patients with no polyp or no label are 'pretrain': unsupervised (DenseCL / CycleGAN) only.
  * The script REFUSES to overwrite an existing split unless --force is passed.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261023
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
ROOT = Path(__file__).resolve().parent.parent
PROC, MAN = ROOT / "data" / "processed", ROOT / "data" / "manifests"
HEADER = ("downloadServerUrl=https://public.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet\n"
          "includeAnnotation=true\nnoOfRetry=4\nmanifestVersion=3.0\nListOfSeriesToDownload=\n")


def assign(ids, rng):
    ids = np.array(sorted(ids)); rng.shuffle(ids)
    n = len(ids); n_tr = int(round(FRACTIONS["train"] * n)); n_va = int(round(FRACTIONS["val"] * n))
    return {**{i: "train" for i in ids[:n_tr]}, **{i: "val" for i in ids[n_tr:n_tr + n_va]},
            **{i: "test" for i in ids[n_tr + n_va:]}}


def main(force=False):
    out = PROC / "cohort_split.csv"
    if out.exists() and not force:
        sys.exit(f"{out} already exists - the split is frozen. Use --force only if you mean to redo it (and re-log why).")
    pat = pd.read_parquet(PROC / "patients.parquet")
    pol = pd.read_parquet(PROC / "polyps.parquet")
    s2 = pd.read_parquet(PROC / "stage2.parquet")
    ser = pd.read_parquet(PROC / "series.parquet")

    has_nm = set(pol.loc[pol.histology_class == "non_malignant", "patient_id"])
    def ctc_role(r):
        if r.label_status.startswith("CONFLICT"):   return "excluded_conflict"
        if r.label_status == "not_in_any_label_file": return "pretrain"
        if r.n_polyps == 0:                         return "pretrain"       # no polyp found
        if r.n_malignant > 0:                       return "challenge"
        return "benign" if r.patient_id in has_nm else "excluded_other_na_only"
    pat["role"] = pat.apply(ctc_role, axis=1)

    rng = np.random.default_rng(SEED)
    rows = []
    for pid in s2.patient_id: rows.append((pid, "StageII", "malignant", "pool"))
    for r in pat.itertuples():
        lab = {"benign": "benign", "challenge": "malignant"}.get(r.role, "")
        rows.append((r.patient_id, "CTC", lab, r.role))
    df = pd.DataFrame(rows, columns=["patient_id", "source", "label", "pool"])

    split = {}
    split.update(assign(df.loc[df.pool == "pool", "patient_id"], rng))                 # StageII cancers
    split.update(assign(df.loc[df.pool == "benign", "patient_id"], rng))               # CTC benign
    df["split"] = df.patient_id.map(split)
    df.loc[df.pool == "challenge", "split"] = "challenge"
    df.loc[df.pool == "pretrain", "split"] = "pretrain"
    df.loc[df.pool.str.startswith("excluded"), "split"] = "excluded"
    assert df.patient_id.is_unique and df.split.notna().all()

    PROC.mkdir(parents=True, exist_ok=True); MAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ---- download manifests: volume series only (no scouts/topograms) for the supervised cohort --------------
    sup = df[df.split.isin(["train", "val", "test", "challenge"])]
    ctc_v = ser[(~ser.is_localizer) & ser.patient_id.isin(sup.loc[sup.source == "CTC", "patient_id"])]
    s2m = s2.merge(sup[["patient_id", "split"]], on="patient_id")
    dl = pd.concat([
        s2m.assign(source="StageII")[["source", "patient_id", "series_uid", "image_count", "file_size_bytes", "split"]],
        ctc_v.merge(sup[["patient_id", "split"]], on="patient_id").assign(source="CTC")[
            ["source", "patient_id", "series_uid", "image_count", "file_size_bytes", "split"]]])
    dl["GB"] = (dl.file_size_bytes / 1e9).round(3)
    dl.to_csv(MAN / "download_series.csv", index=False)
    (MAN / "stageII.tcia").write_text(HEADER + "\n".join(dl[dl.source == "StageII"].series_uid) + "\n")
    (MAN / "ctc_core.tcia").write_text(HEADER + "\n".join(dl[dl.source == "CTC"].series_uid) + "\n")

    counts = df.groupby(["source", "split"]).size().unstack(fill_value=0)
    meta = {"seed": SEED, "fractions": FRACTIONS, "split_unit": "patient",
            "cohort_split_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "counts": {k: {c: int(v) for c, v in r.items()} for k, r in counts.iterrows()},
            "download_GB": dl.groupby("source").GB.sum().round(1).to_dict(),
            "download_series": dl.groupby("source").size().to_dict()}
    (PROC / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(counts.to_string()); print(json.dumps(meta["download_GB"]), json.dumps(meta["download_series"]))
    print("class by split:\n", df[df.split.isin(["train", "val", "test"])].groupby(["split", "label"]).size().unstack())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)

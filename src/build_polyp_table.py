#!/usr/bin/env python
"""Build polyp-level tables for the TCIA CT Colonography (ACRIN 6664) label audit.

Inputs (found by filename pattern inside --raw, so TCIA's original names work):
  *6-to-9-mm-polyps*.xls        wide polyp sheet, one row per patient
  *large-10-mm-polyps*.xls      wide polyp sheet, one row per patient
  *no-polyp-found*.xls          list of patient IDs with no polyp found
  *COLONOGRAPHY*nbia-digest*.xlsx  CT Colonography study/series metadata, whole collection
  *clinical-data*.tsv           clinical TSV (ID cross-check, and sex)
  *StageII*digest*.xlsx         (optional) NBIA digest of StageII-Colorectal-CT

Outputs (parquet, in --out):
  polyps.parquet    one row per polyp (the reshaped LESION n.x columns)
  patients.parquet  one row per patient: label status, polyp counts, series counts
  series.parquet    one row per DICOM series from the NBIA digest, with position
  stage2.parquet    (if the StageII digest is present) one row per StageII patient/series

Usage:
  python src/build_polyp_table.py --raw data/raw --out data/processed

Lesion feature key (from the "Key to CT Colon XL Lesion Feature Spreadsheet"):
  x.1 segment   x.2 size (mm)   x.3 size source   x.4 removed in pieces   x.5 histology
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Code books, transcribed from the key image
# --------------------------------------------------------------------------
SEGMENT = {1: "Rectum", 2: "Sigmoid", 3: "Descending", 4: "Transverse",
           5: "Ascending", 6: "Cecum"}
SIZE_SOURCE = {1: "Colonoscopy", 2: "Pathology"}
PIECES = {1: "No", 2: "Yes"}
HISTOLOGY = {
    1: "Adenocarcinoma",
    2: "Medullary carcinoma",
    3: "Mucinous carcinoma (colloid type)",
    4: "Signet ring cell carcinoma",
    5: "Squamous cell (epidermoid) carcinoma",
    6: "Adenosquamous carcinoma",
    7: "Small cell carcinoma",
    8: "Undifferentiated carcinoma",
    9: "Carcinoma, NOS",
    10: "Hyperplastic",
    11: "Lipomatous",
    12: "Adenomatous",
    13: "Tubular adenoma",
    14: "Tubulovillous adenoma",
    15: "Villous adenoma",
    16: "Tubulovillous adenoma with dysplasia",
    17: "Normal mucosa",
    88: "Other, specify",
    98: "Not applicable",
}
# Grouping for the benign-vs-malignant question. This is OUR grouping of the
# key's codes, not something the dataset supplies: 1-9 are carcinomas, 10-17
# are non-cancer findings (adenomas are benign but pre-malignant). Confirm it
# with whoever set the task, and edit here if they define it differently.
MALIGNANT_CODES = set(range(1, 10))
NON_MALIGNANT_CODES = set(range(10, 18))

N_FEATURES = 5                      # x.1 .. x.5
FEATURE_NAMES = ["segment_code", "size_mm", "size_source_code",
                 "pieces_code", "histology_code"]

# file-pattern -> short tag used in the output
POLYP_FILES = {
    "6to9mm": "*6-to-9-mm-polyps*.xls",
    "ge10mm": "*large-10-mm-polyps*.xls",
}
NO_POLYP_PATTERN = "*no-polyp-found*.xls"
DIGEST_PATTERN = "*COLONOGRAPHY*nbia-digest*.xlsx"
CLINICAL_PATTERN = "*clinical-data*.tsv"


def find_one(raw: Path, pattern: str) -> Path:
    hits = sorted(raw.glob(pattern))
    if len(hits) != 1:
        sys.exit(f"Expected exactly 1 file matching {pattern!r} in {raw}, "
                 f"found {len(hits)}: {[h.name for h in hits]}")
    return hits[0]


# --------------------------------------------------------------------------
# Polyp sheets: wide -> long
# --------------------------------------------------------------------------
POS_TAG_RE = re.compile(r"Pos\s*>=\s*\d+\s*mm", re.I)


def clean_slice_text(v):
    """144 -> '144', 144.0 -> '144', 'Pos >=10mm  147/275' -> '147/275', NaN -> None."""
    if pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(v)
    s = POS_TAG_RE.sub("", str(v)).strip()
    return s or None


def load_wide_polyp_sheet(path: Path, source: str) -> pd.DataFrame:
    """Reshape one wide polyp sheet to one row per polyp."""
    df = pd.read_excel(path)          # first sheet; header row = row 1
    # Columns: patient id, supine slice, prone slice, then n blocks of 5.
    # Headers are inconsistent ("LESION 1.1", then floats 1.2, 1.3 ...), so
    # we go by position, not name.
    if not str(df.columns[0]).lower().startswith("tcia"):
        raise ValueError(f"{path.name}: first column is {df.columns[0]!r}, "
                         "expected the TCIA number")
    feats = df.iloc[:, 3:]
    if feats.shape[1] % N_FEATURES:
        raise ValueError(f"{path.name}: {feats.shape[1]} lesion columns is not "
                         f"a multiple of {N_FEATURES}")
    if not str(feats.columns[0]).upper().startswith("LESION"):
        raise ValueError(f"{path.name}: column 4 is {feats.columns[0]!r}, "
                         "expected 'LESION 1.1'")
    n_lesions = feats.shape[1] // N_FEATURES

    numeric = feats.apply(pd.to_numeric, errors="coerce")
    lost = feats.notna() & numeric.isna()
    if lost.to_numpy().any():
        raise ValueError(f"{path.name}: non-numeric lesion values: "
                         f"{feats.where(lost).stack().unique()[:5]}")

    arr = numeric.to_numpy(dtype=float).reshape(len(df), n_lesions, N_FEATURES)
    n_pat = len(df)
    long = pd.DataFrame(arr.reshape(n_pat * n_lesions, N_FEATURES),
                        columns=FEATURE_NAMES)
    long.insert(0, "lesion_num", np.tile(np.arange(1, n_lesions + 1), n_pat))
    long.insert(0, "patient_id",
                np.repeat(df.iloc[:, 0].astype(str).str.strip().to_numpy(), n_lesions))
    long.insert(2, "source", source)
    # Slice columns mix numbers (144) and strings ("353/432"); the >=10 mm sheet
    # also has a "Pos >=10mm" tag typed into the supine cell. Normalise to text.
    # NOTE: these are PATIENT-level (all of a patient's polyps), not per polyp:
    # the "/"-separated entries are not guaranteed to line up with lesion order.
    sup = df.iloc[:, 1].map(clean_slice_text)
    pro = df.iloc[:, 2].map(clean_slice_text)
    long["supine_slices_patient"] = np.repeat(sup.to_numpy(dtype=object), n_lesions)
    long["prone_slices_patient"] = np.repeat(pro.to_numpy(dtype=object), n_lesions)
    long["pos_ge10mm_tag"] = np.repeat(
        df.iloc[:, 1].astype(str).str.contains(POS_TAG_RE).to_numpy(), n_lesions)

    # Drop empty lesion slots (patients with fewer polyps than the sheet allows).
    # CAUTION: the sheets pre-fill x.4 ("removed in pieces") with 1 = "No" on
    # EVERY empty slot, so "all blank" never matches. A real polyp has at least
    # one of segment / size / size source / histology filled in; x.4 is ignored.
    real = long[[c for c in FEATURE_NAMES if c != "pieces_code"]].notna().any(axis=1)
    long = long[real].reset_index(drop=True)
    return long


def add_labels(polyps: pd.DataFrame) -> pd.DataFrame:
    p = polyps.copy()
    for col in ["segment_code", "size_source_code", "pieces_code", "histology_code"]:
        p[col] = p[col].astype("Int64")
    p["segment"] = p["segment_code"].map(SEGMENT)
    p["size_source"] = p["size_source_code"].map(SIZE_SOURCE)
    p["removed_in_pieces"] = p["pieces_code"].map(PIECES)
    p["histology"] = p["histology_code"].map(HISTOLOGY)

    def hist_class(code):
        if pd.isna(code):
            return "missing"
        if code in MALIGNANT_CODES:
            return "malignant"
        if code in NON_MALIGNANT_CODES:
            return "non_malignant"
        return "other_or_na"          # 88 Other, 98 Not applicable

    p["histology_class"] = p["histology_code"].map(hist_class)
    p["size_bin"] = pd.cut(p["size_mm"], [0, 5, 9, np.inf],
                           labels=["<=5mm", "6-9mm", ">=10mm"])
    p["polyp_id"] = (p["patient_id"] + "_" + p["source"] + "_L"
                     + p["lesion_num"].astype(str))

    # flags for the audit
    p["flag_incomplete"] = p[FEATURE_NAMES].isna().any(axis=1)   # any of x.1-x.5 missing
    expected = p["source"].map({"6to9mm": (0, 10), "ge10mm": (10, np.inf)})
    # sheet name says 6-9 mm / >=10 mm, but a patient's OTHER polyps may be any size
    p["flag_size_outside_sheet_range"] = [
        (not lo <= s < hi) if pd.notna(s) else False
        for s, (lo, hi) in zip(p["size_mm"], expected)
    ]
    return p


# --------------------------------------------------------------------------
# NBIA digest: series table + per-patient series counts
# --------------------------------------------------------------------------
LOCALIZER_RE = re.compile(r"topo|scout|surview|localiz", re.I)


def _age_years(col: pd.Series) -> pd.Series:
    """NBIA ages look like '059Y' -> 59.0 (missing stays NaN)."""
    return pd.to_numeric(col.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


STAGE2_PATTERN = "*StageII*digest*.xlsx"      # optional: second dataset's NBIA digest


def load_stage2(path: Path) -> pd.DataFrame:
    """StageII-Colorectal-CT digest: one row (= one CT series) per patient."""
    d = pd.read_excel(path)
    return pd.DataFrame({
        "patient_id": d["Patient ID"].astype(str).str.strip(),
        "study_uid": d["Study Instance UID"].astype(str),
        "series_uid": d["Series Instance UID"].astype(str),
        "age_years": _age_years(d["Patient Age"]),
        "sex": d["Patient Sex"],
        "study_date": pd.to_datetime(d["Study Date"].astype(str).str.slice(0, 10), errors="coerce"),
        "manufacturer": d["Manufacturer"],
        "model": d["Manufacturer Model Name"],
        "body_part": d["Body Part Examined"],
        "series_description": d["Series Description"],
        "protocol": d["Protocol Name"],
        "image_count": d["Image Count"],
        "file_size_bytes": d["File Size"],
    })


def load_series(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    s = pd.DataFrame({
        "patient_id": d["Patient ID"].astype(str).str.strip(),
        "study_uid": d["Study Instance UID"].astype(str),
        "series_uid": d["Series Instance UID"].astype(str),
        "series_description": d["Series Description"],
        "series_number": d["Series Number"],
        "image_count": d["Image Count"],
        "file_size_bytes": d["File Size"],
        "manufacturer": d["Manufacturer"],
        "model": d["Manufacturer Model Name"],
        "patient_age": _age_years(d["Patient Age"]),
        "patient_sex": d["Patient Sex"],
        # Some studies carry a placeholder date of 2000-01-01 (masked); treat as missing.
        "study_date": pd.to_datetime(d["Study Date"].astype(str).str.slice(0, 10),
                                     errors="coerce").where(
            lambda x: x != pd.Timestamp("2000-01-01")),
        "submitted": pd.to_datetime(d["Max Submission Timestamp"].astype(str).str.slice(0, 19),
                                    errors="coerce"),
    })
    desc = s["series_description"].fillna("").astype(str)
    # Descriptions are free text, abbreviated or truncated ("- PRON", "- SUPI",
    # "COLON PRNE", "Virt Colon Prn", "COLON SUP"). Series whose description says
    # nothing about position ("COLONOGRAPHY", "AXIAL 3D") stay "unknown": resolve
    # those from the DICOM header (Patient Position) if you need them.
    s["position"] = np.select(
        [desc.str.contains(r"pron|prne|\bprn\b", case=False),
         desc.str.contains(r"supi|\bsup\b", case=False)],
        ["prone", "supine"], default="unknown")
    s["is_localizer"] = desc.str.contains(LOCALIZER_RE) | (s["image_count"] <= 5)
    return s


def patient_series_counts(series: pd.DataFrame) -> pd.DataFrame:
    g = series.groupby("patient_id")
    out = pd.DataFrame({
        "n_studies": g["study_uid"].nunique(),
        "n_series": g.size(),
        "n_localizer_series": g["is_localizer"].sum(),
    })
    vol = series[~series["is_localizer"]]
    for pos in ["prone", "supine", "unknown"]:
        out[f"n_{pos}_volume_series"] = (vol[vol["position"] == pos]
                                         .groupby("patient_id").size())
    out = out.fillna(0).astype(int)
    out["has_prone_and_supine"] = ((out["n_prone_volume_series"] > 0)
                                   & (out["n_supine_volume_series"] > 0))
    return out.reset_index()


# --------------------------------------------------------------------------
# Patient table
# --------------------------------------------------------------------------
def build_patients(polyps, ids_by_source, series, clinical_ids) -> pd.DataFrame:
    all_ids = sorted(set(series["patient_id"]) | set().union(*ids_by_source.values()))
    pat = pd.DataFrame({"patient_id": all_ids})
    for src, ids in ids_by_source.items():
        pat[f"in_{src}"] = pat["patient_id"].isin(ids)
    pat["in_clinical"] = pat["patient_id"].isin(clinical_ids)

    n_sources = pat[[f"in_{s}" for s in ids_by_source]].sum(axis=1)
    pat["label_status"] = np.select(
        [n_sources == 0, n_sources > 1],
        ["not_in_any_label_file", "CONFLICT_multiple_label_files"],
        default="labelled")

    g = polyps.groupby("patient_id")
    agg = pd.DataFrame({
        "n_polyps": g.size(),
        "n_malignant": g["histology_class"].apply(lambda x: (x == "malignant").sum()),
        "n_non_malignant": g["histology_class"].apply(lambda x: (x == "non_malignant").sum()),
        "max_size_mm": g["size_mm"].max(),
        "n_ge10mm": g["size_mm"].apply(lambda x: (x >= 10).sum()),
        "n_6to9mm": g["size_mm"].apply(lambda x: x.between(6, 9).sum()),
        "n_le5mm": g["size_mm"].apply(lambda x: (x <= 5).sum()),
    })
    pat = pat.merge(agg, how="left", left_on="patient_id", right_index=True)
    # patients in the no-polyp file have 0 polyps; patients in no file stay NaN (unknown)
    count_cols = ["n_polyps", "n_malignant", "n_non_malignant",
                  "n_ge10mm", "n_6to9mm", "n_le5mm"]
    known = pat["label_status"] != "not_in_any_label_file"
    pat.loc[known, count_cols] = pat.loc[known, count_cols].fillna(0)
    pat[count_cols] = pat[count_cols].astype("Int64")
    pat["has_malignant"] = pat["n_malignant"] > 0
    pat = pat.merge(patient_series_counts(series), how="left", on="patient_id")
    return pat


# --------------------------------------------------------------------------
# QC report
# --------------------------------------------------------------------------
def report(polyps, patients, series, ids_by_source, clinical_ids):
    line = "-" * 60
    print(line, "QC REPORT", line, sep="\n")
    print(f"patients in digest: {series['patient_id'].nunique()}   "
          f"in clinical TSV: {len(clinical_ids)}")
    for src, ids in ids_by_source.items():
        print(f"  {src:<14} {len(ids):>4} patients")
    print(f"polyp rows: {len(polyps)} across {polyps['patient_id'].nunique()} patients")
    print("\nlabel_status:\n" + patients["label_status"].value_counts().to_string())
    conflicts = patients.loc[patients["label_status"].str.startswith("CONFLICT"),
                             "patient_id"].tolist()
    if conflicts:
        print("\n!! patients listed in more than one label file (resolve manually):")
        for c in conflicts:
            print("   ", c)
    print("\nhistology_class:\n" + polyps["histology_class"].value_counts().to_string())
    print("\nhistology (label):\n" + polyps["histology"].value_counts(dropna=False).to_string())
    print("\nsize_bin:\n" + polyps["size_bin"].value_counts(dropna=False).to_string())
    print("\nsegment:\n" + polyps["segment"].value_counts(dropna=False).to_string())
    print(f"\nincomplete polyp rows: {int(polyps['flag_incomplete'].sum())}")
    print(f"polyps whose size is outside their sheet's range: "
          f"{int(polyps['flag_size_outside_sheet_range'].sum())} "
          "(expected: patient's other polyps; check if a surprise)")
    unk = polyps[(polyps["histology_code"].notna()) & polyps["histology"].isna()]
    if len(unk):
        print(f"!! {len(unk)} histology codes not in the key: "
              f"{sorted(unk['histology_code'].unique())}")
    print("\npolyps per labelled patient:\n"
          + patients.loc[patients["n_polyps"].notna(), "n_polyps"]
          .value_counts().sort_index().to_string())
    ok = patients["has_prone_and_supine"].fillna(False)
    print(f"\npatients with prone+supine volume series in digest: {int(ok.sum())} "
          f"of {len(patients)}")
    print(f"labelled patients missing from clinical TSV: "
          f"{int((~patients.loc[patients['label_status'] != 'not_in_any_label_file', 'in_clinical']).sum())}")
    print(line)


# --------------------------------------------------------------------------
def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=root / "data" / "raw")
    ap.add_argument("--out", type=Path, default=root / "data" / "processed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wide = {}
    for tag, pat in POLYP_FILES.items():
        wide[tag] = load_wide_polyp_sheet(find_one(args.raw, pat), tag)
    polyps = add_labels(pd.concat(wide.values(), ignore_index=True))

    # patient lists per source (includes patients whose polyp rows were all empty)
    ids_by_source = {
        tag: set(pd.read_excel(find_one(args.raw, pat)).iloc[:, 0].astype(str).str.strip())
        for tag, pat in POLYP_FILES.items()
    }
    ids_by_source["no_polyp"] = set(
        pd.read_excel(find_one(args.raw, NO_POLYP_PATTERN)).iloc[:, 0].astype(str).str.strip())

    series = load_series(find_one(args.raw, DIGEST_PATTERN))
    clin = pd.read_csv(find_one(args.raw, CLINICAL_PATTERN), sep="\t",
                       usecols=["Blinded_ID", "a0e11"])
    clin["Blinded_ID"] = clin["Blinded_ID"].astype(str).str.strip()
    clinical_ids = set(clin["Blinded_ID"])

    patients = build_patients(polyps, ids_by_source, series, clinical_ids)
    # demographics: age + sex from the NBIA digest (sex often blank), sex also from the clinical TSV
    demo = (series.groupby("patient_id")
            .agg(age_years=("patient_age", "first"), sex_digest=("patient_sex", "first"))
            .reset_index())
    patients = patients.merge(demo, on="patient_id", how="left").merge(
        clin.rename(columns={"Blinded_ID": "patient_id", "a0e11": "sex_clinical"}),
        on="patient_id", how="left")

    polyps.to_parquet(args.out / "polyps.parquet", index=False)
    patients.to_parquet(args.out / "patients.parquet", index=False)
    series.to_parquet(args.out / "series.parquet", index=False)
    print(f"wrote polyps ({len(polyps)}), patients ({len(patients)}), "
          f"series ({len(series)}) -> {args.out}")

    s2_files = sorted(args.raw.glob(STAGE2_PATTERN))
    if s2_files:
        stage2 = load_stage2(s2_files[0])
        stage2.to_parquet(args.out / "stage2.parquet", index=False)
        print(f"wrote stage2 ({len(stage2)} patients) -> {args.out}")
    else:
        print("note: no StageII digest found in --raw; stage2.parquet not written")
    report(polyps, patients, series, ids_by_source, clinical_ids)


if __name__ == "__main__":
    main()

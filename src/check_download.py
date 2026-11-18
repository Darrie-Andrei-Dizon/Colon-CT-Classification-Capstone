"""Check that downloaded series are complete, whatever the folder layout.
    python src/check_download.py --source StageII --root D:\\tcia\\stageII
    python src/check_download.py --source CTC --root D:\\tcia\\ctc
Finds every folder holding .dcm files, reads the SeriesInstanceUID from the first file (header only),
and compares file count and total size with frozen/download_series.csv.
Needs:  pip install pydicom pandas"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pydicom

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["StageII", "CTC"])
ap.add_argument("--root", required=True)
ap.add_argument("--csv", default=str(Path(__file__).resolve().parent / "frozen" / "download_series.csv"))
a = ap.parse_args()

root = Path(a.root)
if not root.exists(): sys.exit(f"{root} does not exist")
dl = pd.read_csv(a.csv); dl = dl[dl.source == a.source]

# group .dcm files by folder, then identify each folder by the UID inside its first file
by_dir = defaultdict(list)
for f in root.rglob("*.dcm"): by_dir[f.parent].append(f)
print(f"found {sum(map(len, by_dir.values()))} .dcm files in {len(by_dir)} folders")
found = defaultdict(lambda: [0, 0])          # uid -> [files, bytes]
for d, files in by_dir.items():
    try: uid = str(pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=["SeriesInstanceUID"]).SeriesInstanceUID)
    except Exception as e: print("unreadable:", files[0], e); continue
    found[uid][0] += len(files); found[uid][1] += sum(x.stat().st_size for x in files)

rows = []
for r in dl.itertuples():
    n, size = found.get(r.series_uid, [0, 0])
    status = ("missing" if n == 0 else "short" if n < r.image_count else "extra" if n > r.image_count else
              "size?" if abs(size - r.file_size_bytes) > 0.02 * r.file_size_bytes else "ok")
    rows.append((r.patient_id, r.series_uid, r.image_count, n, status))
out = pd.DataFrame(rows, columns=["patient_id", "series_uid", "expected", "found", "status"])
print(out.status.value_counts().to_string())
print(f"\n{(out.status == 'ok').sum()} of {len(out)} {a.source} series complete")
unexpected = set(found) - set(dl.series_uid)
if unexpected: print(f"{len(unexpected)} downloaded series are not in the manifest (ignored)")
bad = out[out.status != "ok"]
if len(bad):
    bad.to_csv(root / "problems.csv", index=False); print(bad.head(15).to_string(index=False))
    print(f"\nFull list: {root / 'problems.csv'}")

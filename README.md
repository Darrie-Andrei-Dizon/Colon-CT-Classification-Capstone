# Deep Learning Classification of Benign and Malignant Colon Lesions on CT: Reproducibility Package

This folder contains the code, frozen data split and result tables for the capstone study. Every number in Chapter 4 is produced by a script in `src/`, run in the order below, from the raw files of The Cancer Imaging Archive (TCIA). The CT images themselves are not included, because TCIA licenses them separately; step 1 explains how to download them.

## What is in this folder

```
src/                 all scripts (one per step; each file starts with a short description and example commands)
src/frozen/          the frozen patient-level split, the list of series to download, and TCIA download manifests
data/raw/            TCIA label files for CT Colonography (polyp sheets, clinical data, NBIA digest)
data/processed/      tables made by the scripts (split, slab lists, DICOM headers, colon masks)
data/external/       HQColon expert colon masks (optional check, step 6)
notebooks/           exploratory data analysis (tabular and image)
reports/tables/      every result table (CSV)
reports/figures/     every result figure (PNG); reports/figures/paper/ holds the figures used in the manuscript
requirements.txt     Python packages
```

Large files (the downloaded scans, prepared scans and trained models) are kept outside this folder, on a separate drive. The commands below use `D:\tcia` for scans and `D:\runs` for trained models; change these paths to any folder with enough space.

## Requirements

- **Python** 3.11 (3.10 also works).
- **Disk space:** about 70 GB for the TCIA downloads, about 15 GB for prepared scans, and about 15 GB for the optional MSD Colon positive control.
- **GPU:** an NVIDIA GPU is needed for training (steps 7 to 10). The study used Kaggle's free NVIDIA T4. Data preparation (steps 1 to 6) runs on an ordinary laptop CPU.
- **NBIA Data Retriever** (free, from TCIA) to download the CT scans.

## Step 0. Set up the environment

Run these commands in PowerShell from this folder:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the third line only on a machine with an NVIDIA GPU. On a CPU-only machine, skip it; `requirements.txt` then installs the CPU build of PyTorch. On Kaggle, PyTorch is already installed; run only the last line.

All later commands assume the environment is active (`.\.venv\Scripts\Activate.ps1`), or replace `python` with `.\.venv\Scripts\python.exe`.

## Step 1. Get the data

| Dataset | Role in the study | Where to get it |
| --- | --- | --- |
| StageII-Colorectal-CT (230 patients) | Malignant class | TCIA, doi:10.7937/p5k5-tg43 |
| CT Colonography, ACRIN 6664 (825 patients) | Benign class and the 3 held-out cancers | TCIA, doi:10.7937/K9/TCIA.2015.NWTESAY1 |
| CT Colonography label files (polyp sheets, clinical data, NBIA digest) | Labels | TCIA collection page; already in `data/raw/` |
| HQColon expert colon masks (optional) | Check of the automatic colon selection | Finocchiaro et al. (2026), doi:10.1038/s41597-025-06518-z; place the zip files in `data/external/hqcolon/` |
| Medical Segmentation Decathlon, Task 10 Colon (optional) | Positive control | Downloaded automatically in step 11 |
| RadImageNet weights (optional) | Medical-image pretraining comparison | `ResNet50.pt` from https://huggingface.co/Lab-Rasool/RadImageNet |

**Download the scans.** Open `src/frozen/stageII.tcia` and `src/frozen/ctc_core.tcia` in the NBIA Data Retriever and save them to `D:\tcia\stageII` and `D:\tcia\ctc`. These manifests list exactly the 456 series used in the study (about 69 GB). Then check that nothing is missing:

```
python src/check_download.py --source StageII --root D:\tcia\stageII
python src/check_download.py --source CTC --root D:\tcia\ctc
```

## Step 2. Labels and the patient split (already done; for checking only)

The split is frozen in `src/frozen/cohort_split.csv` (seed 20261023), and `data/processed/cohort_split.csv` is the same file. To rebuild it from the label files:

```
python src/build_polyp_table.py
python src/make_split.py
```

`make_split.py` refuses to overwrite an existing split unless `--force` is given, so the frozen split cannot be changed by accident. Its SHA-256 checksum is stored in `src/frozen/split_meta.json`.

## Step 3. Read DICOM headers and prepare the scans

```
python src/extract_headers.py --stageII D:\tcia\stageII --ctc D:\tcia\ctc
python src/preprocess.py --out D:\tcia\processed
```

This step puts every scan in a common orientation, averages it into 5 mm slabs, resamples it to 1.5 mm pixels, removes the table and padding, and saves one `.npy` file per series. Add `--limit 6` to try a few scans first.

## Step 4. Select the colon region

```
python src/select_colon.py --fast --device gpu --download src/frozen/download_series.csv
```

This step uses TotalSegmentator (fast model). Use `--device cpu` if there is no GPU; it is slower but gives the same result. It writes `data/processed/colon_slices.csv` and the colon masks in `data/processed/colon/`, and it can be stopped and resumed.

## Step 5. Source checks and the brightness-only baseline (Sections 4.2.4, 4.4.1, 4.4.2)

```
python src/source_check_colon.py --npy-dir D:\tcia\processed --label original
python src/evaluate.py --baseline intensity --name intensity_lr
python src/acquisition_check.py
python src/acquisition_match.py --npy-dir D:\tcia\processed
```

## Step 6. Check the colon selection against expert masks (optional; Section 4.2.5)

```
python src/check_colon_vs_hqcolon.py
```

## Step 7. CycleGAN translation (Section 4.3.1.1; GPU)

Version 1 (whole slab):

```
python src/cyclegan.py train --npy-dir D:\tcia\processed --out D:\runs\cyclegan_v1 --iters 40000
python src/cyclegan.py translate --npy-dir D:\tcia\processed --checkpoint D:\runs\cyclegan_v1\last.pt --direction AtoB --out-dir D:\tcia\translated_v1
python src/translate_qc.py --orig-dir D:\tcia\processed --trans-dir D:\tcia\translated_v1 --label cyclegan_v1
```

Version 2 (colon region only), used in experiments C and E:

```
python src/cyclegan.py train --npy-dir D:\tcia\processed --out D:\runs\cyclegan_v2 --iters 40000 --mask-dir data/processed/colon --dilate 4
python src/cyclegan.py translate --npy-dir D:\tcia\processed --checkpoint D:\runs\cyclegan_v2\last.pt --direction AtoB --out-dir D:\tcia\translated_v2 --mask-dir data/processed/colon
python src/translate_qc.py --orig-dir D:\tcia\processed --trans-dir D:\tcia\translated_v2 --label cyclegan_v2
python src/source_check_colon.py --npy-dir D:\tcia\translated_v2 --orig-dir D:\tcia\processed --label cyclegan_v2
```

## Step 8. DenseCL pretraining (Section 4.3.1.2; GPU)

```
python src/densecl.py --arch densenet121 --npy-dir D:\tcia\processed --out D:\runs\dcl_densenet_raw
python src/densecl.py --arch resnet50 --npy-dir D:\tcia\processed --out D:\runs\dcl_resnet_raw
python src/densecl.py --arch densenet121 --npy-dir D:\tcia\translated_v2 --slices data/processed/colon_slices_cyclegan_v2.csv --out D:\runs\dcl_densenet_cyc
python src/densecl.py --arch resnet50 --npy-dir D:\tcia\translated_v2 --slices data/processed/colon_slices_cyclegan_v2.csv --out D:\runs\dcl_resnet_cyc
```

Each run saves `encoder.pt` in its folder.

## Step 9. Classifiers, experiments A to E and CO (Section 4.3.2; GPU)

| Experiment | Command |
| --- | --- |
| A | `python src/classify.py --arch densenet121 --npy-dir D:\tcia\processed --out D:\runs\A_densenet` |
| B | `python src/classify.py --arch resnet50 --npy-dir D:\tcia\processed --out D:\runs\B_resnet` |
| C | `python src/classify.py --arch densenet121 --npy-dir D:\tcia\translated_v2 --slices data/processed/colon_slices_cyclegan_v2.csv --out D:\runs\C_densenet` (and `--arch resnet50`, `--out D:\runs\C_resnet`) |
| D | `python src/classify.py --arch densenet121 --npy-dir D:\tcia\processed --encoder D:\runs\dcl_densenet_raw\encoder.pt --out D:\runs\D_densenet` (and the ResNet-50 version with `dcl_resnet_raw`) |
| E | `python src/classify.py --arch densenet121 --npy-dir D:\tcia\translated_v2 --slices data/processed/colon_slices_cyclegan_v2.csv --encoder D:\runs\dcl_densenet_cyc\encoder.pt --out D:\runs\E_densenet` (and the ResNet-50 version with `dcl_resnet_cyc`) |
| CO | `python src/colon_only_classify.py --masks data/processed/colon --arch densenet121 --npy-dir D:\tcia\processed --out D:\runs\CO_densenet` (and `--arch resnet50`, `--out D:\runs\CO_resnet`) |

Each run writes `best.pt` and `preds.csv` to its folder. If a session time limit stops a run, run the same command again to resume. Then evaluate each run at patient level, and rank the 3 held-out cancers:

```
python src/evaluate.py --pred D:\runs\A_densenet\preds.csv --name A_densenet
python src/ctc_only_rank.py A=D:\runs\A_densenet\preds.csv B=D:\runs\B_resnet\preds.csv D=D:\runs\D_resnet\preds.csv
```

Repeat `evaluate.py` for every run folder, and pass every run to `ctc_only_rank.py`.

## Step 10. Grad-CAM (Section 4.3.2.2)

```
python src/gradcam.py --arch densenet121 --checkpoint D:\runs\A_densenet\best.pt --npy-dir D:\tcia\processed --out D:\runs\A_densenet\gradcam_test
python src/gradcam.py --arch densenet121 --checkpoint D:\runs\A_densenet\best.pt --npy-dir D:\tcia\processed --split-name challenge --out D:\runs\A_densenet\gradcam_challenge
```

Repeat for the D and C runs. Each output folder contains `gradcam_summary.csv` (the share of heat inside the colon) and the heat-map images.

## Step 11. Analyses inside CT Colonography (Section 4.4.4)

```
python src/within_ctc_cv.py --model size
python src/within_ctc_cv.py --model intensity --npy-dir D:\tcia\processed
python src/extract_embeddings.py --arch resnet50 --npy-dir D:\tcia\processed --out D:\runs\emb_imagenet_resnet.parquet
python src/extract_embeddings.py --arch resnet50 --encoder D:\runs\dcl_resnet_raw\encoder.pt --npy-dir D:\tcia\processed --out D:\runs\emb_dcl_resnet.parquet
python src/within_ctc_cv.py --model embeddings --embeddings D:\runs\emb_imagenet_resnet.parquet --name imagenet_resnet --npy-dir D:\tcia\processed
python src/within_ctc_cv.py --model embeddings --embeddings D:\runs\emb_dcl_resnet.parquet --name dcl_resnet --npy-dir D:\tcia\processed
python src/within_ctc_finetune.py --arch densenet121 --npy-dir D:\tcia\processed --out D:\runs\wctc_dn_imagenet
python src/within_ctc_finetune.py --arch densenet121 --npy-dir D:\tcia\processed --encoder D:\runs\dcl_densenet_raw\encoder.pt --out D:\runs\wctc_dn_densecl
```

Polyp-slice analysis, with the RadImageNet comparison:

```
python src/within_ctc_polyp.py map --k 1
python src/extract_embeddings.py --arch resnet50 --npy-dir D:\tcia\processed --slices data\processed\colon_slices_polyp_k1.csv --out D:\runs\emb_imagenet_resnet_polyp_k1.parquet
python src/extract_embeddings.py --arch resnet50 --encoder D:\runs\dcl_resnet_raw\encoder.pt --npy-dir D:\tcia\processed --slices data\processed\colon_slices_polyp_k1.csv --out D:\runs\emb_dcl_resnet_polyp_k1.parquet
python src/extract_embeddings.py --arch resnet50 --imagenet-weights radimagenet=D:\weights\radimagenet\ResNet50.pt --npy-dir D:\tcia\processed --out D:\runs\emb_rin_resnet.parquet
python src/extract_embeddings.py --arch resnet50 --imagenet-weights radimagenet=D:\weights\radimagenet\ResNet50.pt --npy-dir D:\tcia\processed --slices data\processed\colon_slices_polyp_k1.csv --out D:\runs\emb_rin_resnet_polyp_k1.parquet
python src/within_ctc_polyp.py compare --embeddings imagenet_whole=D:\runs\emb_imagenet_resnet.parquet imagenet_polyp=D:\runs\emb_imagenet_resnet_polyp_k1.parquet dcl_whole=D:\runs\emb_dcl_resnet.parquet dcl_polyp=D:\runs\emb_dcl_resnet_polyp_k1.parquet rin_whole=D:\runs\emb_rin_resnet.parquet rin_polyp=D:\runs\emb_rin_resnet_polyp_k1.parquet
```

## Step 12. Positive control on MSD Colon (Section 4.4.5)

```
python src/msd_colon.py download
python src/msd_colon.py prepare
python src/msd_colon.py colon --device gpu --fast
python src/msd_colon.py export
```

`download` saves about 6 GB to `D:\msd` and resumes if the connection drops. Then train on a GPU (in the study, Kaggle):

```
python src/classify.py --arch densenet121 --npy-dir D:\msd\processed --slices data/processed/msd_classify_slices.csv --split data/processed/msd_classify_split.csv --out D:\runs\msd_dn_imagenet
python src/classify.py --arch resnet50 --npy-dir D:\msd\processed --slices data/processed/msd_classify_slices.csv --split data/processed/msd_classify_split.csv --out D:\runs\msd_rn_imagenet
python src/msd_colon.py evaluate --pred D:\runs\msd_dn_imagenet\preds.csv --name densenet121_imagenet
python src/msd_colon.py evaluate --pred D:\runs\msd_rn_imagenet\preds.csv --name resnet50_imagenet
```

## Where each result appears

| Manuscript | Result file(s) in `reports/` |
| --- | --- |
| Table 2, Section 4.2.1 (datasets) | `tables/img_02_geometry.csv`, `tables/img_03_acquisition.csv` |
| Table 3, Section 4.2.2 (cohort and split) | `data/processed/cohort_split.csv`, `src/frozen/split_meta.json` |
| Table 4, Section 4.2.4 (source prediction) | `tables/img_07_source_prediction.csv` |
| Section 4.2.5 (colon selection check) | `tables/colon_vs_hqcolon_summary.csv` |
| Table 5, Figure 12 (CycleGAN) | `tables/translate_qc_cyclegan_v1_*.csv`, `tables/translate_qc_cyclegan_v2_*.csv`, `tables/source_check_colon.csv` |
| Table 6 (classification) | `tables/eval_<experiment>.csv`, `tables/ctc_only_rank.csv` |
| Table 7, Figure 13 (Grad-CAM) | `gradcam_summary.csv` in each run folder |
| Table 8 (inside CT Colonography) | `tables/within_ctc_cv_*.csv`, `tables/within_ctc_polyp_compare_k1.csv` |
| Table 9 (positive control) | `tables/msd_eval.csv` |
| Scan-matched check (limitations) | `tables/acq_match_*.csv` |
| Figure 14 (AUC summary) | `figures/paper/fig_auc_summary.png` |

## Notes on reproducibility

- **One seed throughout:** 20261023, for the split, model initialization and data order. Results can still differ slightly between GPU types because some GPU operations are not deterministic.
- **The split is fixed by patient,** so prone and supine scans of one patient are always in the same set. The 3 CT Colonography cancers are never used for training or tuning.
- **Every script prints what it writes,** and most can be resumed after a stop.
- **Hardware used in the study:** a Windows laptop (CPU) for steps 1 to 6, 11 and 12 (data preparation), and a Kaggle NVIDIA T4 GPU for steps 7 to 10 and the MSD training.

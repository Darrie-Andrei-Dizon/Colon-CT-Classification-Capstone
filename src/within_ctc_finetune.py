"""Fine-tuned network inside CT Colonography only: advanced adenoma vs the rest, cross-validated by patient.

    python within_ctc_finetune.py --arch densenet121 --npy-dir <scans> --slices colon_slices.csv --split cohort_split.csv \
        --patients within_ctc_patients.csv --out <folder> [--encoder <DenseCL encoder.pt>] [--task adv_vs_other_benign]

Why: in the main design every malignant scan is StageII and every benign scan is CT Colonography, so a network can
score 1.00 by recognising the dataset.  Here every scan comes from ONE dataset with ONE preparation and protocol, so
that shortcut cannot work.  Whether a fine-tuned network can separate advanced adenoma from other patients is the
fairest answer this study can give to "can the network see a lesion at all".  It is exploratory: 95 labelled patients.

Data: the 95 labelled CT Colonography patients (train, val and test of the frozen split pooled; the 3 cancers and the
excluded patients are NOT used), grouped by worst lesion in within_ctc_patients.csv (written by within_ctc_cv.py):
    adv_vs_other_benign       advanced adenoma vs everything else      (28 vs 67)   default
    adv_vs_nonneoplastic      advanced adenoma vs non-neoplastic       (28 vs 25)
    adenoma_vs_nonneoplastic  any adenoma vs non-neoplastic            (70 vs 25)
Every kept colon slab of a patient gets the patient's label (lesion slices are mostly unknown), so most slabs of a
positive patient do not show the lesion.  That label noise is a limit of this analysis.

Cross-validation: stratified 5-fold by PATIENT, repeated --repeats times with fixed seeds.  Nothing is tuned on the
held-out fold: the schedule is fixed (3 epochs head only, then the last block unfrozen for 7 epochs, same as classify.py)
and the LAST epoch is used.  Every patient gets an out-of-fold score per repeat: for each scan the mean of its 5 highest
slab scores, averaged over the patient's scans.  Training slabs are sampled so every patient counts equally and the two
classes are balanced.  Reported: AUC with a bootstrap 95% interval over patients, sensitivity and specificity at 0.5 with
Clopper-Pearson intervals (same summary as within_ctc_cv.py).  No permutation p-value (it would need hundreds of retrainings).

Outputs in --out: oof_<task>.csv (patient_id, group, label, repeat, fold, score), done.json (finished folds, resumable:
run the same command again), summary_<task>.csv, run_config.json.
Needs: torch torchvision numpy pandas scikit-learn scipy (backbones.py, colon_dataset.py, classify.py, within_ctc_cv.py alongside)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd
from backbones import Encoder, load_pretrained
from classify import Classifier, set_phase
from within_ctc_cv import TASKS, summarise

ROOT = Path(__file__).resolve().parent.parent


class Slabs(cd.ColonSlices):
    """Some patients' slabs from a ColonSlices dataset, with the within-CT-Colonography label instead of benign/malignant."""

    def __init__(self, base, patients, ymap, augment):
        m = base.rows.patient_id.isin(patients).values
        self.rows = base.rows[m].reset_index(drop=True); self._r = self.rows.to_dict("records")
        self.channels, self.size, self.augment = base.channels, base.size, augment
        self.y = self.rows.patient_id.map(ymap).astype(int).values

    def __getitem__(self, i):
        it = super().__getitem__(i); it["label"] = torch.tensor(int(self.y[i])); return it

    def sampler(self, seed):
        """Every patient counts equally; the two classes together have equal weight."""
        n_slab = self.rows.groupby("patient_id").slab_idx.transform("size").values
        pat_y = pd.Series(self.y, index=self.rows.patient_id.values).groupby(level=0).first()
        n_cls = pat_y.value_counts(); w = 1.0 / (n_slab * pd.Series(self.y).map(n_cls).values)
        g = torch.Generator(); g.manual_seed(seed)
        return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(self), replacement=True, generator=g)


@torch.no_grad()
def patient_scores(model, ds, dev, batch, workers, amp, k=5):
    model.eval(); out = []
    for b in DataLoader(ds, batch, shuffle=False, num_workers=workers):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out.append(torch.sigmoid(model(b["image"].to(dev)).float()).cpu().numpy())
    r = ds.rows.assign(score=np.concatenate(out))
    per_scan = r.groupby(["patient_id", "series_uid"]).score.apply(lambda s: s.nlargest(k).mean())
    return per_scan.groupby("patient_id").mean()


def train_fold(a, base, train_p, ymap, seed, dev, amp):
    cd.seed_everything(seed)
    tr = Slabs(base, train_p, ymap, augment=True)
    dl = DataLoader(tr, a.batch, sampler=tr.sampler(seed), drop_last=True, num_workers=a.workers, worker_init_fn=cd.seed_worker)
    enc = load_pretrained(a.arch, a.encoder) if a.encoder else Encoder(a.arch, a.imagenet_weights)
    model = Classifier(enc, a.dropout).to(dev); crit = nn.BCEWithLogitsLoss()     # sampler already balances the classes
    scaler = torch.amp.GradScaler(enabled=amp); total = a.freeze_epochs + a.ft_epochs
    for ep in range(total):
        ph = 1 if ep < a.freeze_epochs else 2
        if ep in (0, a.freeze_epochs):
            opt = torch.optim.Adam(set_phase(model, ph), lr=a.lr_head if ph == 1 else a.lr_ft, weight_decay=1e-4)
        model.train(); model.enc.eval() if ph == 1 else None; tl, n = 0.0, 0
        for it, b in enumerate(dl, 1):
            x, y = b["image"].to(dev, non_blocking=True), b["label"].float().to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x).float(), y)
            if not torch.isfinite(loss): raise SystemExit(f"loss is {loss.item()}: stopping (try --no-amp)")
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item(); n += 1
            if a.max_iters and it >= a.max_iters: break
        print(f"    epoch {ep + 1}/{total} (phase {ph}) train loss {tl / max(n, 1):.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["densenet121", "resnet50"], required=True)
    ap.add_argument("--npy-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--patients", default=str(ROOT / "reports/tables/within_ctc_patients.csv"), help="patient_id, group (from within_ctc_cv.py)")
    ap.add_argument("--task", choices=list(TASKS), default="adv_vs_other_benign")
    ap.add_argument("--encoder", default=None, help="DenseCL encoder.pt; default = ImageNet weights")
    ap.add_argument("--imagenet-weights", default="imagenet")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--freeze-epochs", type=int, default=3); ap.add_argument("--ft-epochs", type=int, default=7)
    ap.add_argument("--lr-head", type=float, default=1e-3); ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda" and not a.no_amp
    pt = pd.read_csv(a.patients); posg, negg = TASKS[a.task]; pt = pt[pt.group.isin(posg + negg)].reset_index(drop=True)
    ymap = dict(zip(pt.patient_id, pt.group.isin(posg).astype(int))); y = pt.group.isin(posg).astype(int).values
    base = cd.ColonSlices(["train", "val", "test"], a.npy_dir, a.slices, a.split, channels=3, size=a.size, augment=False, sources=["CTC"])
    have = set(base.rows.patient_id); pt = pt[pt.patient_id.isin(have)].reset_index(drop=True); y = pt.group.isin(posg).astype(int).values
    print(f"{a.task}: {int(y.sum())} positive and {int((1 - y).sum())} negative patients | {a.arch} | "
          f"init {'DenseCL ' + a.encoder if a.encoder else a.imagenet_weights} | device {dev.type} | amp {amp}", flush=True)

    oof_path, done_path = out / f"oof_{a.task}.csv", out / "done.json"
    done = set(map(tuple, json.load(open(done_path)))) if done_path.exists() else set()
    rows = pd.read_csv(oof_path).to_dict("records") if oof_path.exists() else []
    t0 = time.time(); last = 0.0
    for rep in range(a.repeats):
        for fold, (tr, te) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=cd.SEED + rep).split(pt.patient_id, y)):
            if (rep, fold) in done: continue
            if a.max_hours and last and (time.time() - t0) / 3600 + 1.1 * last > a.max_hours:
                print("stopping before the time limit; run the same command again to continue"); return
            t1 = time.time(); print(f"repeat {rep + 1}/{a.repeats} fold {fold + 1}/{a.folds}: train {len(tr)} patients, test {len(te)}", flush=True)
            model = train_fold(a, base, set(pt.patient_id[tr]), ymap, cd.SEED + 100 * rep + fold, dev, amp)
            sc = patient_scores(model, Slabs(base, set(pt.patient_id[te]), ymap, augment=False), dev, a.batch, a.workers, amp)
            for pid, s in sc.items(): rows.append(dict(patient_id=pid, group=pt.group[pt.patient_id == pid].iloc[0], label=ymap[pid], repeat=rep, fold=fold, score=float(s)))
            done.add((rep, fold)); pd.DataFrame(rows).to_csv(oof_path, index=False); json.dump(sorted(done), open(done_path, "w"))
            last = (time.time() - t1) / 3600; print(f"  fold done in {last * 60:.1f} min", flush=True)

    O = pd.DataFrame(rows); P = O.groupby("patient_id").agg(label=("label", "first"), score=("score", "mean")).reset_index()
    per_rep = [roc_auc_score(g.label, g.score) for _, g in O.groupby("repeat") if g.label.nunique() == 2]
    r = summarise(P.label.values.astype(int), P.score.values, per_rep, np.random.default_rng(cd.SEED))
    r.update(task=a.task, arch=a.arch, init="DenseCL" if a.encoder else a.imagenet_weights, repeats=a.repeats, folds=a.folds)
    S = pd.DataFrame([r]); S.to_csv(out / f"summary_{a.task}.csv", index=False)
    json.dump(vars(a), open(out / "run_config.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print("\nOut-of-fold result (patient level, average over repeats):")
    print(S[["task", "arch", "init", "n_pos", "n_neg", "auc", "auc_lo", "auc_hi", "auc_repeat_mean", "auc_repeat_sd", "sensitivity", "specificity", "TP", "FN", "TN", "FP"]].round(3).to_string(index=False))
    print("AUC 0.5 = no signal.  Polyp-size reference (within_ctc_cv.py --model size) is about 0.9 and is partly circular.")


if __name__ == "__main__":
    main()

"""Benign-versus-malignant classifier for experiments A to E of Table 1 (manuscript 3.5 stages 4 to 7).

    python classify.py --arch densenet121 --npy-dir <scans> --slices colon_slices.csv --split cohort_split.csv --out <run folder> [options]

Table 1                                   --npy-dir            --encoder (DenseCL weights)      --arch
  A  raw CT, standard pretrained, DenseNet     original scans       (none)                           densenet121
  B  raw CT, standard pretrained, ResNet       original scans       (none)                           resnet50
  C  CycleGAN CT, standard pretrained          translated scans     (none)                           either
  D  raw CT, DenseCL                           original scans       dcl_raw_<arch>/encoder.pt        either
  E  CycleGAN CT, DenseCL                      translated scans     dcl_cyc_<arch>/encoder.pt        either
Pretrained weights: ImageNet (torchvision, needs internet once) unless --encoder is given, which loads the DenseCL
encoder (the DenseCL run itself started from ImageNet, so the order is ImageNet -> DenseCL -> here, as in 3.5 stage 4).

Model: encoder -> global average pooling -> dropout -> one dense layer -> sigmoid (probability of MALIGNANT).
Loss: binary cross-entropy with a positive-class weight (benign slabs / malignant slabs in the training set).
Schedule (stage 6): the whole backbone is frozen for the first --freeze-epochs epochs (only the head trains, lr-head); then
only the LAST block (DenseNet: dense block 4; ResNet: layer 4) is unfrozen together with the head at the lower --lr-ft
for --ft-epochs more epochs.  Batch-norm statistics stay frozen throughout.  Best epoch = lowest validation loss.
Data: colon_dataset.ColonSlices (frozen patient-level split, window -400..400 HU, no resizing, small rotation/scale
augmentation, NO flip).  Train on 'train', pick the epoch on 'val', then predict val, test and the 3-patient challenge set.

Outputs: <out>/best.pt, <out>/last.pt (resumable: run the same command again), <out>/train_log.csv, <out>/run_config.json,
         <out>/preds.csv   (patient_id, series_uid, slab_idx, score, split)  ->  python src/evaluate.py --pred <out>/preds.csv --name <exp>
Needs: torch torchvision numpy pandas scikit-learn (and colon_dataset.py, backbones.py in the same folder)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd
from backbones import Encoder, load_pretrained

ROOT = Path(__file__).resolve().parent.parent


class Classifier(nn.Module):
    def __init__(self, enc, dropout=0.3):
        super().__init__()
        self.enc, self.drop, self.fc = enc, nn.Dropout(dropout), nn.Linear(enc.out_ch, 1)

    def forward(self, x):
        return self.fc(self.drop(self.enc(x).mean((2, 3)))).squeeze(1)       # logit; sigmoid gives P(malignant)


def set_phase(model, phase):
    """phase 1: only the head trains.  phase 2: head + last block of the backbone."""
    for p in model.enc.parameters(): p.requires_grad = False
    if phase == 2:
        for m in model.enc.last_block():
            for p in m.parameters(): p.requires_grad = True
    for p in model.fc.parameters(): p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def predict(model, ds, dev, batch, workers, amp):
    model.eval(); dl = DataLoader(ds, batch, shuffle=False, num_workers=workers); out = []
    for b in dl:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out.append(torch.sigmoid(model(b["image"].to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["densenet121", "resnet50"], required=True)
    ap.add_argument("--npy-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--encoder", default=None, help="DenseCL encoder.pt (experiments D, E); default = ImageNet weights")
    ap.add_argument("--imagenet-weights", default="imagenet", help="imagenet | none | path to a torchvision state dict (offline machines)")
    ap.add_argument("--freeze-epochs", type=int, default=3); ap.add_argument("--ft-epochs", type=int, default=7)
    ap.add_argument("--lr-head", type=float, default=1e-3); ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--seed", type=int, default=cd.SEED)
    ap.add_argument("--max-hours", type=float, default=0); ap.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    a = ap.parse_args()

    cd.seed_everything(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda" and not a.no_amp
    mk = lambda sp, aug: cd.ColonSlices(sp, a.npy_dir, a.slices, a.split, channels=3, size=a.size, augment=aug)
    tr, va = mk("train", True), mk("val", False)
    dl = DataLoader(tr, a.batch, shuffle=True, drop_last=True, num_workers=a.workers, worker_init_fn=cd.seed_worker, persistent_workers=a.workers > 0)
    n_mal = int((tr.rows.label == "malignant").sum()); n_ben = int((tr.rows.label == "benign").sum())
    pos_w = torch.tensor([n_ben / max(n_mal, 1)], device=dev)
    enc = load_pretrained(a.arch, a.encoder) if a.encoder else Encoder(a.arch, a.imagenet_weights)
    model = Classifier(enc, a.dropout).to(dev); crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    total = a.freeze_epochs + a.ft_epochs; scaler = torch.amp.GradScaler(enabled=amp)
    log_path, start, best = out / "train_log.csv", 0, float("inf"); opt, phase = None, 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev); model.load_state_dict(ck["model"]); start, best = ck["epoch"], ck["best"]
        print(f"resumed after epoch {start}")
    elif log_path.exists(): log_path.unlink()
    print(f"device {dev.type} | amp {amp} | {a.arch} | init {'DenseCL ' + a.encoder if a.encoder else a.imagenet_weights} | train slabs {len(tr)} "
          f"(malignant {n_mal}, benign {n_ben}, pos_weight {pos_w.item():.2f}) | val slabs {len(va)}", flush=True)
    t_start = time.time(); hours0 = json.load(open(out / "run_config.json")).get("hours_so_far", 0) if (out / "run_config.json").exists() else 0
    for ep in range(start, total):
        ph = 1 if ep < a.freeze_epochs else 2
        if ph != phase:                                   # (re)build the optimiser when the phase changes
            params = set_phase(model, ph); phase = ph
            opt = torch.optim.Adam(params, lr=a.lr_head if ph == 1 else a.lr_ft, weight_decay=1e-4)
            if (out / "last.pt").exists() and start and torch.load(out / "last.pt", map_location=dev).get("phase") == ph:
                opt.load_state_dict(torch.load(out / "last.pt", map_location=dev)["opt"])
        model.train(); model.enc.eval() if ph == 1 else None; t0 = time.time(); tl, n = 0.0, 0
        for it, b in enumerate(dl, 1):
            x, y = b["image"].to(dev, non_blocking=True), b["label"].float().to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at epoch {ep + 1} iteration {it}: the starting weights or the data contain NaN/inf. Stopping. Check the encoder with: python -c \"import torch; sd=torch.load('<encoder.pt>'); print([k for k,v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()][:5])\"")
            tl += loss.item(); n += 1
            if it % 50 == 0: print(f"epoch {ep + 1}/{total} (phase {ph}) it {it}/{len(dl)} | loss {tl / n:.4f} | {it * a.batch / (time.time() - t0):.0f} img/s", flush=True)
            if a.max_iters and it >= a.max_iters: break
        sv = predict(model, va, dev, a.batch, a.workers, amp); yv = (va.rows.label == "malignant").astype(int).values
        pv = np.clip(sv, 1e-6, 1 - 1e-6); vl = float(-(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean())
        vauc = roc_auc_score(yv, sv) if len(np.unique(yv)) == 2 else float("nan")
        row = f"{ep + 1},{ph},{tl / max(n, 1):.4f},{vl:.4f},{vauc:.4f},{time.time() - t0:.0f}\n"
        with open(log_path, "a") as f:
            if not log_path.exists() or log_path.stat().st_size == 0: f.write("epoch,phase,train_loss,val_loss,val_auc_slab,seconds\n")
            f.write(row)
        if vl < best: best = vl; torch.save(model.state_dict(), out / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep + 1, best=best, phase=ph), out / "last.pt.tmp"); (out / "last.pt.tmp").replace(out / "last.pt")
        hours = hours0 + (time.time() - t_start) / 3600
        json.dump(dict(vars(a), hours_so_far=round(hours, 3), epochs_done=ep + 1, best_val_loss=best, pos_weight=pos_w.item()), open(out / "run_config.json", "w"), indent=1)
        print(f"epoch {ep + 1} done | train loss {tl / max(n, 1):.4f} | val loss {vl:.4f} | val slab AUC {vauc:.3f} | {time.time() - t0:.0f} s", flush=True)
        if a.max_hours and ep + 1 < total and hours + 1.1 * (time.time() - t0) / 3600 > a.max_hours:
            print("stopping before the time limit; run the same command again to continue"); return
    if not (out / "best.pt").exists(): torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)); frames = []
    for sp in ["val", "test", "challenge"]:
        try: ds = mk(sp, False)
        except ValueError as e: print(f"{sp}: {e}"); continue
        f = cd.predictions_frame(ds, predict(model, ds, dev, a.batch, a.workers, amp)); f["split"] = sp; frames.append(f)
        print(f"predicted {sp}: {len(ds)} slabs, {ds.rows.patient_id.nunique()} patients")
    pd.concat(frames).to_csv(out / "preds.csv", index=False); print("wrote", out / "preds.csv")


if __name__ == "__main__":
    main()

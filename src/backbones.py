"""The two backbones of the study (DenseNet-121, ResNet-50), shared by DenseCL pretraining and the classifiers.

    from backbones import Encoder, load_pretrained

Encoder(arch, weights)
    input : (B, 3, H, W) float tensor in [-1, 1]  (what colon_dataset produces with channels=3)
    output: feature map (B, C, h, w)   (C = 1024 for DenseNet-121, 2048 for ResNet-50; h = w = H / 32)
    ImageNet normalisation is done inside, so the same encoder object works for DenseCL, the classifier and Grad-CAM.

weights: "imagenet" (downloads torchvision weights; needs internet), "none" (random start, for tests),
         or a path to a torchvision state-dict file (.pth) for offline machines (e.g. an uploaded Kaggle dataset),
         or "radimagenet=<path>" for RadImageNet weights (Mei et al., 2022; PyTorch files from
         https://huggingface.co/Lab-Rasool/RadImageNet : ResNet50.pt, DenseNet121.pt), e.g.
         --imagenet-weights radimagenet=D:\\weights\\radimagenet\\ResNet50.pt  in classify.py and extract_embeddings.py.
         RadImageNet was trained on 1.35 million CT, MRI and ultrasound images; the same ImageNet normalisation is used
         (as the weights' model card says).

Batch-norm: by default the batch-norm layers keep their ImageNet statistics and are not updated (freeze_bn=True).
Reasons: DenseCL's two-branch design leaks information through batch statistics on a single GPU, and small batches make
batch statistics noisy.  The batch-norm scale and shift parameters are still trained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCHS = ("densenet121", "resnet50")


class Encoder(nn.Module):
    def __init__(self, arch, weights="imagenet", freeze_bn=True):
        super().__init__()
        import torchvision.models as tvm
        self.arch, self.freeze_bn = arch, freeze_bn
        if arch == "densenet121":
            w = tvm.DenseNet121_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
            m = tvm.densenet121(weights=w)
        elif arch == "resnet50":
            w = tvm.ResNet50_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
            m = tvm.resnet50(weights=w)
        else:
            raise ValueError(f"arch must be one of {ARCHS}")
        if isinstance(weights, str) and weights.lower().startswith("radimagenet="):
            load_radimagenet(m, arch, weights.split("=", 1)[1])
        elif weights not in ("imagenet", "none"):
            m.load_state_dict(torch.load(weights, map_location="cpu"))
        if arch == "densenet121":
            self.body, self.out_ch = m.features, 1024
            self._last = [self.body.denseblock4, self.body.norm5]          # the "last dense block" (manuscript 3.5)
        else:
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
            self.out_ch = 2048
            self._last = [self.body[7]]                                    # layer4 = the last residual stage
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = ((x + 1) / 2 - self.mean) / self.std
        f = self.body(x)
        return F.relu(f) if self.arch == "densenet121" else f

    def last_block(self):
        return self._last

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()
        return self


HEADS = {"resnet50": "fc.", "densenet121": "classifier."}
PREFIXES = ("module.", "model.", "backbone.", "encoder.", "base_model.", "net.")


def _unwrap(obj):
    if isinstance(obj, nn.Module): obj = obj.state_dict()
    for k in ("state_dict", "model_state_dict", "model", "net"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], (dict, nn.Module)): return _unwrap(obj[k])
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def load_radimagenet(m, arch, path):
    """Load a RadImageNet PyTorch file (e.g. Lab-Rasool/RadImageNet ResNet50.pt / DenseNet121.pt) into torchvision model m.

    The file's key names are not documented, so: (1) match by name after removing common prefixes; if that fails,
    (2) match by layer order, requiring every tensor shape to agree.  The classifier head (fc / classifier) is not
    needed and is left as is.  Refuses to load unless every backbone tensor is matched."""
    src = _unwrap(torch.load(path, map_location="cpu", weights_only=False))
    tgt = {k: v for k, v in m.state_dict().items() if not k.startswith(HEADS[arch])}
    clean = {}
    for k, v in src.items():
        for p in PREFIXES:
            while k.startswith(p): k = k[len(p):]
        if arch == "densenet121" and not k.startswith(("features.", "classifier.")): k = "features." + k
        clean[k] = v
    by_name = {k: clean[k] for k in tgt if k in clean and clean[k].shape == tgt[k].shape}
    how = "by name"
    if len(by_name) < len(tgt):                                     # fall back to layer order + shape
        skeys = [k for k in src if not k.endswith("num_batches_tracked")]
        tkeys = [k for k in tgt if not k.endswith("num_batches_tracked")]
        if len(skeys) < len(tkeys) or any(src[s].shape != tgt[t].shape for s, t in zip(skeys, tkeys)):
            raise SystemExit(f"{path}: could not match the RadImageNet weights to torchvision {arch} "
                             f"(by name {len(by_name)}/{len(tgt)} tensors; by order the shapes differ). Check that the file is the {arch} one.")
        by_name = {t: src[s] for s, t in zip(skeys, tkeys)}; how = "by layer order"
        for k in tgt:
            if k.endswith("num_batches_tracked"): by_name[k] = tgt[k]
    missing, _ = m.load_state_dict(by_name, strict=False)
    missing = [k for k in missing if not k.startswith(HEADS[arch])]
    if missing: raise SystemExit(f"{path}: {len(missing)} backbone tensors not loaded, e.g. {missing[:3]}")
    bad = [k for k, v in by_name.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad: raise SystemExit(f"{path}: weights contain NaN/inf, e.g. {bad[:3]}")
    rv = max(float(v.max()) for k, v in by_name.items() if k.endswith("running_var"))
    print(f"RadImageNet {arch}: {len(by_name)} tensors loaded {how} from {path} | largest batch-norm running_var {rv:.3g}")
    if rv > 1e4:
        print("  note: some batch-norm variances are very large (a known quirk of these weights, RadImageNet GitHub issue #20). "
              "Batch-norm statistics stay frozen; if features or losses turn NaN or extreme, report it rather than tuning around it.")


def load_pretrained(arch, ckpt, freeze_bn=True):
    """Encoder with DenseCL weights (the encoder.pt written by densecl.py)."""
    e = Encoder(arch, "none", freeze_bn)
    e.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return e

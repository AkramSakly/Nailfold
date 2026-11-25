"""
train_compare.py  --  PARTIE A : CLASSIFICATION (dataset Density)
==================================================================
Entraine et COMPARE les 3 modeles de l'article :
  - CE-NFCNet : Cascade EfficientNet (le modele propose)
  - SE-NFCNet : Single EfficientNet (transfer learning simple)
  - CC-NFCNet : Cascade CNN (CNN entraine de zero, sans EfficientNet)

Produit un tableau comparatif (comme le Tableau 6 de l'article) :
  Accuracy, Precision, Recall, F1, ROC-AUC pour chaque modele.

Evaluation : validation croisee groupee par sujet (StratifiedGroupKFold),
pour eviter la fuite de donnees (un sujet n'est jamais a la fois en
entrainement et en validation).

NOTE IMPORTANTE (a ecrire dans le rapport) :
  L'article fait un VRAI cascade avec Dataset A (dermatomycosis + tortuosity)
  puis Dataset B (Normal/Abnormal). Le dataset Density ne contient PAS ces
  sous-classes. On adapte donc :
   - CE : on fine-tune les stages profonds d'EfficientNet (transfer learning
          a 2 niveaux : ImageNet -> stages 6-7-8, puis tete) = esprit cascade
   - SE : on fine-tune seulement la tete + stage 8 (transfer simple, 1 etape)
   - CC : un petit CNN entraine entierement de zero (pas d'ImageNet)
  La difference CE vs SE vs CC reflete donc bien la logique de l'article :
  profondeur du transfer learning croissante.

UTILISATION :
  python classification/train_compare.py --index data/index.csv --folds 2 --epochs 60
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG = 224


# ---------------------------------------------------------------- DATA
def build_tf(train):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG, IMG)),
            transforms.RandomRotation(72),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DS(Dataset):
    def __init__(self, rows, train):
        self.rows, self.tf = rows, build_tf(train)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), \
               torch.tensor(int(r["label"]))


# ---------------------------------------------------------------- MODELES
def make_efficientnet_head(in_feats, dropout=0.3):
    return nn.Sequential(nn.Dropout(dropout), nn.BatchNorm1d(in_feats),
                         nn.Linear(in_feats, 2))


class EfficientNetClassifier(nn.Module):
    """Sert pour CE et SE. Le mode change quels stages sont entrainables."""
    def __init__(self, mode):
        super().__init__()
        b = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = b.features
        self.avgpool = b.avgpool
        f = b.classifier[1].in_features
        self.pre = nn.Dropout2d(0.3)
        self.head = make_efficientnet_head(f)
        # geler tout le backbone
        for p in self.features.parameters():
            p.requires_grad = False
        if mode == "CE":
            stages = [6, 7, 8]      # transfer profond (esprit cascade)
        elif mode == "SE":
            stages = [8]            # transfer simple (1 etape, peu de couches)
        else:
            raise ValueError(mode)
        for idx in stages:
            for p in self.features[idx].parameters():
                p.requires_grad = True
    def forward(self, x):
        x = self.features(x); x = self.pre(x); x = self.avgpool(x)
        return self.head(torch.flatten(x, 1))


class SmallCNN(nn.Module):
    """CC-NFCNet : CNN entraine entierement de zero (pas d'ImageNet)."""
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.body = nn.Sequential(
            block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 2))
    def forward(self, x):
        x = self.body(x); x = self.pool(x)
        return self.head(torch.flatten(x, 1))


def build_model(name):
    if name in ("CE", "SE"):
        return EfficientNetClassifier(name)
    return SmallCNN()


# ---------------------------------------------------------------- ENTRAINEMENT
def class_weights(rows, device):
    c = np.bincount([r["label"] for r in rows], minlength=2)
    c = np.clip(c, 1, None)
    return torch.tensor(c.sum() / (2.0 * c), dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device)).cpu()); Y.append(y)
    logits = torch.cat(L); y = torch.cat(Y).numpy()
    p = torch.softmax(logits, 1).numpy()[:, 1]
    return y, (p >= 0.5).astype(int), p


def metrics(y, yp, p):
    m = {"accuracy": accuracy_score(y, yp),
         "precision": precision_score(y, yp, zero_division=0),
         "recall": recall_score(y, yp, zero_division=0),
         "f1": f1_score(y, yp, zero_division=0)}
    try: m["roc_auc"] = roc_auc_score(y, p)
    except ValueError: m["roc_auc"] = float("nan")
    return m


def train_fold(name, tr_rows, va_rows, args, device):
    tr_ds, va_ds = DS(tr_rows, True), DS(va_rows, False)
    drop = (len(tr_ds) % args.batch_size) == 1
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=drop)
    va = DataLoader(va_ds, batch_size=args.batch_size, num_workers=0)
    model = build_model(name).to(device)
    lr = 1e-4 if name in ("CE", "SE") else 1e-3   # CC part de zero -> lr plus grand
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr_rows, device))
    best, best_state, patience = 1e9, None, args.patience
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                v += crit(model(x), y).item() * x.size(0)
        v /= len(va.dataset)
        if v < best - 1e-4:
            best, best_state, patience = v, \
                {k: val.cpu().clone() for k, val in model.state_dict().items()}, \
                args.patience
        else:
            patience -= 1
        if patience <= 0:
            break
    if best_state: model.load_state_dict(best_state)
    return metrics(*evaluate(model, va, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs_compare")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")

    rows = []
    with open(args.index) as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"]); rows.append(r)
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    folds = min(args.folds, len(set(groups)))

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                               random_state=args.seed)
    splits = list(skf.split(np.zeros(len(rows)), labels, groups))

    all_results = {}
    for name in ["CE", "SE", "CC"]:
        print(f"\n########## Modele {name}-NFCNet ##########")
        fold_m = []
        for fi, (tr, va) in enumerate(splits, 1):
            print(f"  pli {fi}/{folds} ...", end="", flush=True)
            m = train_fold(name, [rows[i] for i in tr],
                           [rows[i] for i in va], args, device)
            print(" " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
            fold_m.append(m)
        agg = {}
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = np.array([m[k] for m in fold_m], dtype=float)
            agg[k] = float(np.nanmean(vals))
        all_results[name] = agg

    # ---- tableau comparatif ----
    print("\n\n===== TABLEAU COMPARATIF (style Tableau 6 de l'article) =====")
    print(f"{'Modele':<14}{'Accuracy':<11}{'Precision':<11}"
          f"{'Recall':<9}{'F1':<8}{'ROC_AUC'}")
    for name in ["CE", "SE", "CC"]:
        a = all_results[name]
        print(f"{name+'-NFCNet':<14}{a['accuracy']:<11.3f}{a['precision']:<11.3f}"
              f"{a['recall']:<9.3f}{a['f1']:<8.3f}{a['roc_auc']:.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "comparaison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultats sauvegardes dans {args.outdir}/comparaison.json")


if __name__ == "__main__":
    main()
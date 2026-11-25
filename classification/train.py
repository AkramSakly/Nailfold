"""
train.py
--------
Trains CE-NFCNet (EfficientNet-B0 + transfer learning) to classify
nailfold capillary images as Normal vs Abnormal, following:
  Ebadi Jalal et al. (2025), Scientific Reports 15:2068.

IMPORTANT - how this matches / adapts the paper
------------------------------------------------
Faithful to the paper:
  * EfficientNet-B0 backbone, ImageNet-pretrained (transfer learning)
  * Paper's regularization head: Dropout(0.3) -> GAP -> Dropout(0.3)
    -> BatchNorm -> Dense  (see model.py)
  * Only the deeper stages are fine-tuned; backbone is otherwise frozen
  * Paper's augmentations (rotation up to 72 deg, H/V flip, brightness/contrast)
  * Adam optimizer, early stopping on validation loss
  * Metrics: Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrix

Adapted because your dataset differs from the paper's:
  * The paper's full *cascade* (Step A on dermatomycosis+tortuosity, then
    Step B on Normal vs Abnormal) needs sub-class labels you don't have, and
    a fixed 12-image test set that would be statistically meaningless on
    30 images / 4 subjects. So instead we use the SAME backbone + head + the
    paper's Step-B fine-tuning (stages 7,8) and evaluate with SUBJECT-GROUPED
    k-fold cross-validation. Grouping by subject prevents leakage (the same
    person's frames never appear in both train and validation), which is the
    correct protocol for such a small clinical dataset.
  * Class imbalance (13 Normal / 17 Abnormal) is handled with a class-weighted
    cross-entropy loss.

Usage:
  python prepare_data.py --dossier Density --out data
  python classification/train.py --index data/index.csv --folds 4 --epochs 100

Outputs (in --outdir, default 'runs'):
  * best model weights per fold
  * metrics.json (per-fold + mean +/- std)
  * confusion_matrix.png, roc_curves.png
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from dataset import NFCDataset
from model import CENFCNet


# ---------------------------------------------------------------------- #
def load_index(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def make_loaders(train_rows, val_rows, batch_size):
    train_ds = NFCDataset(train_rows, train=True)
    val_ds = NFCDataset(val_rows, train=False)
    # drop_last=True avoids a final batch of size 1, which would break the
    # BatchNorm layer in the classifier head during training.
    drop_last = (len(train_ds) % batch_size) == 1
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, drop_last=drop_last)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2)
    return train_dl, val_dl


def class_weights(rows, device):
    counts = np.bincount([r["label"] for r in rows], minlength=2)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (2.0 * counts)        # inverse-frequency weighting
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=1).numpy()[:, 1]   # P(Abnormal)
    y_pred = (probs >= 0.5).astype(int)
    return y_true, y_pred, probs


def compute_metrics(y_true, y_pred, probs):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC needs both classes present in y_true
    try:
        m["roc_auc"] = roc_auc_score(y_true, probs)
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


# ---------------------------------------------------------------------- #
def train_one_fold(train_rows, val_rows, args, device, fold_idx):
    train_dl, val_dl = make_loaders(train_rows, val_rows, args.batch_size)

    model = CENFCNet(num_classes=2, dropout=0.3, pretrained=True).to(device)
    # Paper Step B: fine-tune stages 7 and 8 only (backbone otherwise frozen).
    model.set_trainable_stages([7, 8])
    print(f"  [fold {fold_idx}] trainable params: {model.num_trainable():,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_dl.dataset)

        # validation loss
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                vrun += criterion(model(x), y).item() * x.size(0)
        val_loss = vrun / len(val_dl.dataset)

        improved = val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch % 5 == 0 or improved:
            print(f"    epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}{'  *' if improved else ''}")

        if patience_left <= 0:
            print(f"    early stopping at epoch {epoch}")
            break

    # restore best and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred, probs = evaluate(model, val_dl, device)
    metrics = compute_metrics(y_true, y_pred, probs)

    # save weights
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(args.outdir, f"fold{fold_idx}_best.pt"))

    return metrics, y_true, y_pred, probs


# ---------------------------------------------------------------------- #
def plot_results(all_true, all_pred, all_probs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)

    # Confusion matrix (pooled over folds)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Abnormal"])
    ax.set_yticklabels(["Normal", "Abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (pooled CV)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve (pooled)
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (pooled CV)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "roc_curves.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=7)   # paper Step B
    ap.add_argument("--batch_size", type=int, default=8) # paper Step B
    ap.add_argument("--lr", type=float, default=1e-4)    # paper Step B
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = load_index(args.index)
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    n_subjects = len(set(groups))

    # Can't have more folds than subjects (subject-grouped CV)
    folds = min(args.folds, n_subjects)
    if folds < args.folds:
        print(f"Reducing folds to {folds} (only {n_subjects} subjects).")

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                               random_state=args.seed)

    fold_metrics = []
    all_true, all_pred, all_probs = [], [], []

    X_dummy = np.zeros(len(rows))
    for fold_idx, (tr, va) in enumerate(skf.split(X_dummy, labels, groups), 1):
        train_rows = [rows[i] for i in tr]
        val_rows = [rows[i] for i in va]
        tr_subj = sorted({r["subject"] for r in train_rows})
        va_subj = sorted({r["subject"] for r in val_rows})
        print(f"\n=== Fold {fold_idx}/{folds} ===")
        print(f"  train subjects: {tr_subj}  ({len(train_rows)} imgs)")
        print(f"  val   subjects: {va_subj}  ({len(val_rows)} imgs)")

        metrics, yt, yp, pr = train_one_fold(
            train_rows, val_rows, args, device, fold_idx)
        print(f"  fold metrics: " +
              ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        fold_metrics.append(metrics)
        all_true.append(yt); all_pred.append(yp); all_probs.append(pr)

    # Aggregate
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {}
    print("\n=== Cross-validation summary (mean +/- std) ===")
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        mean = np.nanmean(vals); std = np.nanstd(vals)
        summary[k] = {"mean": float(mean), "std": float(std),
                      "per_fold": [float(v) for v in vals]}
        print(f"  {k:10s}: {mean:.3f} +/- {std:.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"per_fold": fold_metrics, "summary": summary}, f, indent=2)

    plot_results(all_true, all_pred, all_probs, args.outdir)
    print(f"\nSaved metrics + plots to '{args.outdir}/'")


if __name__ == "__main__":
    main()
import csv
import os

import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

SUBJECT_TO_FOLD = {"S3": 1, "N1": 2, "S2": 3, "N2": 4}
RUNS_DIR = "runs"
INDEX = os.path.join("data", "index.csv")
CLASSES = ["Normal", "Abnormal"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CENFCNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_feats = backbone.classifier[1].in_features
        self.pre_pool_dropout = nn.Dropout2d(p=dropout)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pre_pool_dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rows = []
with open(INDEX) as f:
    for r in csv.DictReader(f):
        r["label"] = int(r["label"])
        rows.append(r)

model_cache = {}
def get_model(fold):
    if fold not in model_cache:
        path = os.path.join(RUNS_DIR, f"fold{fold}_best.pt")
        m = CENFCNet().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        model_cache[fold] = m
    return model_cache[fold]

print(f"{'Image':<14}{'Reel':<10}{'Predit':<10}{'P(Anormal)':<12}{'OK?'}")
print("-" * 56)

n_ok = 0
results = []
for r in rows:
    fold = SUBJECT_TO_FOLD.get(r["subject"], 1)
    model = get_model(fold)
    img = Image.open(r["path"]).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    correct = (pred == r["label"])
    n_ok += correct
    name = os.path.basename(r["path"])
    print(f"{name:<14}{CLASSES[r['label']]:<10}{CLASSES[pred]:<10}"
          f"{probs[1]:<12.3f}{'oui' if correct else 'NON'}")
    results.append({
        "image": name, "subject": r["subject"],
        "reel": CLASSES[r["label"]], "predit": CLASSES[pred],
        "p_anormal": round(float(probs[1]), 4),
        "correct": correct,
    })

print("-" * 56)
print(f"Accuracy globale : {n_ok}/{len(rows)} = {n_ok/len(rows):.3f}")

out_csv = os.path.join(RUNS_DIR, "predictions_toutes.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "subject", "reel", "predit",
                                      "p_anormal", "correct"])
    w.writeheader()
    w.writerows(results)
print(f"Tableau complet sauvegarde dans : {out_csv}")
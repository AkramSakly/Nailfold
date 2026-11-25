import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

with open("runs_compare/comparaison.json") as f:
    res = json.load(f)

modeles = ["CE", "SE", "CC"]
metriques = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x = np.arange(len(metriques))
w = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
for i, m in enumerate(modeles):
    vals = [res[m][k] for k in metriques]
    bars = ax.bar(x + i * w, vals, w, label=m + "-NFCNet")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + w)
ax.set_xticklabels(metriques)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Comparaison des 3 modeles de classification (dataset Density)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/comparaison_modeles.png", dpi=150)
plt.close(fig)

print("Figure creee : figures/comparaison_modeles.png")
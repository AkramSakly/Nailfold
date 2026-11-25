"""
Prepare_data_dossier.py
-----------------------
Version qui lit directement un DOSSIER d'images deja decompresse
(pas besoin de fichier .zip). Cherche automatiquement le dossier 'DATA'
ou qu'il soit, puis construit data/index.csv.

UTILISATION (dans le terminal, avec le python du venv) :
  python prepare_data.py --dossier Density --out data

Regle d'etiquetage : sujet commencant par 'N' -> Normal (0),
                     sujet commencant par 'S' -> Anormal (1).
"""

import argparse
import csv
import os


def trouver_data(racine):
    """Cherche un dossier nomme 'DATA' (ou contenant des sous-dossiers N/S)."""
    # 1) cherche un dossier exactement nomme DATA
    for dirpath, dirnames, _ in os.walk(racine):
        if os.path.basename(dirpath).upper() == "DATA":
            return dirpath
    # 2) sinon, cherche un dossier qui contient des sous-dossiers N.. / S..
    for dirpath, dirnames, _ in os.walk(racine):
        sous = [d for d in dirnames if d[:1].upper() in ("N", "S")]
        if len(sous) >= 2:
            return dirpath
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default="Density",
                    help="Dossier deja decompresse contenant les images")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data_dir = trouver_data(args.dossier)
    if data_dir is None:
        raise RuntimeError(
            f"Impossible de trouver le dossier d'images sous '{args.dossier}'. "
            "Verifiez que les sous-dossiers sujets (N1, N2, S2, S3...) existent."
        )
    print(f"Dossier d'images trouve : {data_dir}")

    rows = []
    for subject in sorted(os.listdir(data_dir)):
        subj_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        prefix = subject[0].upper()
        if prefix == "N":
            label, name = 0, "Normal"
        elif prefix == "S":
            label, name = 1, "Abnormal"
        else:
            print(f"  [ignore] sujet '{subject}' (prefixe inconnu)")
            continue
        for fname in sorted(os.listdir(subj_path)):
            low = fname.lower()
            if not low.endswith((".jpg", ".jpeg", ".png")):
                continue
            if "_natif" in low:
                continue
            rows.append({
                "path": os.path.abspath(os.path.join(subj_path, fname)),
                "subject": subject,
                "label": label,
                "label_name": name,
            })

    if not rows:
        raise RuntimeError("Aucune image trouvee. Verifiez le contenu du dossier.")

    csv_path = os.path.join(args.out, "index.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "subject", "label", "label_name"])
        w.writeheader()
        w.writerows(rows)

    n_norm = sum(r["label"] == 0 for r in rows)
    n_abn = sum(r["label"] == 1 for r in rows)
    subjects = sorted({r["subject"] for r in rows})
    print("\n=== Resume du jeu de donnees ===")
    print(f"Index ecrit dans : {csv_path}")
    print(f"Total images     : {len(rows)}")
    print(f"Normal (0)       : {n_norm}")
    print(f"Anormal (1)      : {n_abn}")
    print(f"Sujets ({len(subjects)})     : {', '.join(subjects)}")


if __name__ == "__main__":
    main()
"""Fine-tuning de l'encodeur pré-entraîné (Stratégie 2, Personne C).
Folds et format de sortie communs au projet ; compatible avec scripts/ensemble.py.

    python train_tl.py --folds group                          # outputs/tl/
    python train_tl.py --folds group --ckpt none              # ablation : même archi sans pré-entraînement -> outputs/tl_scratch/
    python train_tl.py --folds group --label-frac 0.25 --name tl_25
    python train_tl.py --only-fold 0 --epochs 2               # test rapide
    python train_tl.py --rnafm --ckpt none                    # RNA-FM gelé seul -> outputs/tl_rnafm/
    python train_tl.py --rnafm                                # MLM + RNA-FM -> outputs/tl_mlm_rnafm/
    python -m experiments.transfer.finetune --folds group                          # outputs/tl/
    python -m experiments.transfer.finetune --folds group --ckpt none              # ablation : même archi sans pré-entraînement -> outputs/tl_scratch/
    python -m experiments.transfer.finetune --folds group --label-frac 0.25 --name tl_25
    python -m experiments.transfer.finetune --only-fold 0 --epochs 2               # test rapide

Sorties dans outputs/<name>/ : oof.npy, submission.csv, scores.json, history_fold*.csv, model_fold*.pt
"""
import argparse

from src.transfer.finetune import run
from src.transfer.mlm import DEFAULT_CKPT


def parse_args():
    p = argparse.ArgumentParser()
    # ---- options identiques à experiments/from_scratch/train.py
    p.add_argument("--name", help="dossier de sortie (défaut : tl, ou tl_scratch avec --ckpt none)")
    p.add_argument("--folds", default="kfold", choices=["kfold", "group"])
    p.add_argument("--only-fold", type=int, help="n'entraîner qu'un fold (tests rapides)")
    p.add_argument("--lr", type=float, default=1e-3, help="lr de la tête (et de tout le modèle sans pré-entraînement)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--all-samples", action="store_true",
                   help="entraîner aussi sur les molécules bruitées (SN_filter = 0)")
    # ---- options propres au transfer learning
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="encodeur pré-entraîné, ou 'none'")
    p.add_argument("--lr-enc", type=float, default=1e-4, help="lr de l'encodeur pré-entraîné")
    p.add_argument("--rnafm", action="store_true",
                   help="ajouter les embeddings RNA-FM gelés (lancer extract_rnafm.py avant)")
    p.add_argument("--freeze-epochs", type=int, default=3, help="epochs avec encodeur gelé")
    p.add_argument("--label-frac", type=float, default=1.0, help="fraction des molécules d'entraînement")
    p.add_argument("--sn-weight", action="store_true", help="pondérer la loss par le signal/bruit")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""Extraction des embeddings RNA-FM (modèle de fondation gelé), une seule fois.

    pip install rna-fm
    python extract_rnafm.py               # -> outputs/rnafm/{train,test_107,test_130}.npy
    python extract_rnafm.py --force       # réécrit les fichiers existants

RNA-FM n'est jamais entraîné : simple inférence, sans gradient. On garde la sortie de
la dernière couche (12) : un vecteur de 640 dimensions par nucléotide, stocké en float16
(environ 0,9 Go au total). Les identifiants sont enregistrés à côté (<nom>_ids.npy) pour
vérifier l'alignement avec data/processed au chargement.
"""
import argparse

import numpy as np
import torch

from src import config as C
from src.preprocessing.data import load_processed
from src.transfer.data import PROCESSED, RNAFM_DIM, RNAFM_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 12


def sequences(proc):
    """Séquences A/C/G/U reconstruites à partir du canal « base » des tokens."""
    return ["".join(C.SEQ_VOCAB[t] for t in row) for row in proc["tokens"][:, :, 0]]


@torch.no_grad()
def embed(model, batch_converter, seqs, bs):
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        _, _, toks = batch_converter([(str(j), s) for j, s in enumerate(chunk)])
        rep = model(toks.to(DEVICE), repr_layers=[LAYER])["representations"][LAYER]
        L = len(chunk[0])
        # le tokenizer ajoute des tokens de début / fin : on ne garde que les L nucléotides
        if rep.shape[1] == L + 2:
            rep = rep[:, 1:-1]
        elif rep.shape[1] == L + 1:
            rep = rep[:, 1:]
        if rep.shape[1] != L:
            raise RuntimeError(f"forme inattendue {tuple(rep.shape)} pour des séquences de longueur {L}")
        out.append(rep.float().cpu().numpy().astype("float16"))
        print(f"\r  {min(i + bs, len(seqs))}/{len(seqs)}", end="", flush=True)
    print()
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true", help="réécrire les fichiers existants")
    args = p.parse_args()
    try:
        import fm
    except ImportError:
        raise SystemExit("Package RNA-FM absent : pip install rna-fm")

    RNAFM_DIR.mkdir(parents=True, exist_ok=True)
    model, alphabet = fm.pretrained.rna_fm_t12()      # télécharge les poids au premier appel
    model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    print(f"RNA-FM chargé sur {DEVICE}")

    for name in PROCESSED:
        path = RNAFM_DIR / f"{name}.npy"
        if path.exists() and not args.force:
            print(f"{name} : déjà extrait ({path})")
            continue
        proc = load_processed(name)
        seqs = sequences(proc)
        print(f"{name} : {len(seqs)} molécules de longueur {len(seqs[0])}")
        E = embed(model, batch_converter, seqs, args.batch_size)
        if E.shape != (len(seqs), len(seqs[0]), RNAFM_DIM):
            raise RuntimeError(f"{name} : forme {E.shape}, attendu ({len(seqs)}, {len(seqs[0])}, {RNAFM_DIM})")
        np.save(path, E)
        np.save(RNAFM_DIR / f"{name}_ids.npy", proc["ids"].astype(str))
        print(f"  -> {path}  {E.shape}  {E.nbytes / 1e6:.0f} Mo")


if __name__ == "__main__":
    main()

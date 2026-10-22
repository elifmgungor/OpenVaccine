"""Entrées de la stratégie 2, lues dans data/processed/ (écrit par scripts/preprocess.py).

Mêmes données que experiments/from_scratch/train.py. Comme dans le reste du projet, un batch ne contient
qu'une seule longueur : un « groupe » = un dict de tableaux de même longueur L.
"""
import numpy as np

from src import config as C
from src.preprocessing.data import inputs_of, load_processed, select

TL_KEYS = ["tokens", "feats", "bpps", "pairs"]   # onehot inutile : couche Embedding
PROCESSED = ("train", "test_107", "test_130")

# Embeddings RNA-FM (extract_rnafm.py), rangés dans outputs/ donc sur le Drive avec Colab
RNAFM_DIR = C.OUTPUT_DIR / "rnafm"
RNAFM_DIM = 640


def load_rnafm(name, ids):
    """Embeddings RNA-FM (N, L, 640) en float16, alignés sur les molécules de `ids`."""
    path, ids_path = RNAFM_DIR / f"{name}.npy", RNAFM_DIR / f"{name}_ids.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer d'abord `python extract_rnafm.py`")
    saved = np.load(ids_path, allow_pickle=False).astype(str)
    if not np.array_equal(saved, np.asarray(ids).astype(str)):
        raise ValueError(f"{name} : l'ordre des molécules diffère de data/processed, relancer "
                         "`python extract_rnafm.py --force`")
    return np.load(path)


def load_inputs(name, rnafm=False):
    """Entrées du modèle pour 'train', 'test_107' ou 'test_130'.
    rnafm=True ajoute la clé 'rnafm' (N, L, 640) : embeddings du modèle de fondation gelé."""
    proc = load_processed(name)
    X = select(inputs_of(proc), slice(None), TL_KEYS)
    if rnafm:
        X["rnafm"] = load_rnafm(name, proc["ids"])
    return X


def sn_weight(signal_to_noise):
    """Poids optionnel par molécule (option --sn-weight)."""
    return (0.5 * np.log(np.clip(signal_to_noise, 0, None) + 1.1)).astype("float32")


def n_batches(groups, bs):
    return sum(-(-len(g["tokens"]) // bs) for g in groups)


def iterate_batches(groups, bs, shuffle=False, rng=None, device="cpu"):
    """Batches de longueur homogène convertis en tenseurs torch.
    Sans shuffle, l'ordre de sortie est celui des groupes puis des lignes."""
    import torch
    jobs = []
    for gi, g in enumerate(groups):
        n = len(g["tokens"])
        order = rng.permutation(n) if shuffle else np.arange(n)
        jobs += [(gi, order[i:i + bs]) for i in range(0, n, bs)]
    if shuffle:
        jobs = [jobs[j] for j in rng.permutation(len(jobs))]
    for gi, idx in jobs:
        yield {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(device)
               for k, v in groups[gi].items()}

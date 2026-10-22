"""Test rapide de la stratégie 2 sur un petit sous-ensemble des données prétraitées.

    python -m tests.smoke_transfer

Moins d'une minute sur CPU. N'écrit rien dans outputs/ (fichier temporaire uniquement).
Chaque étape affiche OK ou s'arrête avec un message explicite.
"""
import os
import sys
import tempfile
import time

T0 = time.time()
_step = [0]


def step(msg):
    _step[0] += 1
    print(f"\n[{_step[0]}] {msg}")


def ok(msg):
    print(f"    OK  {msg}")


def fail(msg):
    print(f"    ÉCHEC  {msg}")
    sys.exit(1)


# ------------------------------------------------------------------ 1
step("Imports et dépendances")
try:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch
    import keras
except ImportError as e:
    fail(f"{e} -> pip install -r requirements.txt (et torch)")
try:
    import tabulate  # noqa: F401  (utilisé par compare.py via DataFrame.to_markdown)
except ImportError:
    print("    ATTENTION  tabulate absent : pip install tabulate (requis par compare.py)")
from src import config as C
from src.evaluation.metrics import mcrmse
from src.evaluation.submission import to_submission
from src.preprocessing.data import load_processed
from src.transfer.data import RNAFM_DIM, RNAFM_DIR, load_inputs, iterate_batches
from src.transfer.model import RNAEncoder, MLMModel, RegModel, MASK_IDS
from src.transfer.mlm import apply_mlm_mask, mlm_loss
from src.transfer.finetune import mcrmse_loss, predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ok(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
   f"torch {torch.__version__} | keras {keras.__version__} (backend {keras.backend.backend()})")
ok(f"device : {DEVICE}")

# ------------------------------------------------------------------ 2
step("Données prétraitées")
for name in ("train", "test_107", "test_130"):
    if not (C.PROCESSED_DIR / f"{name}.npz").exists():
        fail(f"{C.PROCESSED_DIR / name}.npz absent -> python -m scripts.preprocess")
if not (C.DATA_DIR / "sample_submission.csv").exists():
    fail(f"{C.DATA_DIR / 'sample_submission.csv'} absent (requis par to_submission)")
train = load_processed("train")
needed = {"ids", "targets", "sn_filter", "signal_to_noise", "fold_kfold", "fold_group"}
if needed - set(train):
    fail(f"clés manquantes dans train.npz : {needed - set(train)}")
N = 32
X = {k: v[:N] for k, v in load_inputs("train").items()}
Y = train["targets"][:N]
T107 = {k: v[:8] for k, v in load_inputs("test_107").items()}
T130 = {k: v[:8] for k, v in load_inputs("test_130").items()}
ok(f"train.npz : {len(train['ids'])} molécules ; sous-ensemble de test : {N}")
ok("shapes : " + ", ".join(f"{k} {v.shape}" for k, v in X.items()) + f", targets {Y.shape}")

# ------------------------------------------------------------------ 3
step("Batching (longueur homogène, conversion torch)")
b = next(iterate_batches([X], 8, device=DEVICE))
assert b["tokens"].shape == (8, 107, 3) and b["bpps"].shape == (8, 107, 107)
ok(f"batch : tokens {tuple(b['tokens'].shape)} {b['tokens'].dtype}, feats {b['feats'].dtype}")

# ------------------------------------------------------------------ 4
step("Masquage MLM (proportions et absence de fuite)")
torch.manual_seed(0)
mb, lab = apply_mlm_mask(b)
frac = (lab["seq"] != -100).float().mean().item()
assert 0.05 < frac < 0.3, f"proportion masquée anormale : {frac:.3f}"
joint = lab["struct"] != -100
assert joint.any(), "aucune structure masquée"
assert (mb["tokens"][..., 1][joint] == MASK_IDS[1]).all(), "token [MASK] structure absent"
assert mb["adj"][joint].abs().sum() == 0, "fuite : adjacence non effacée"
assert mb["feats"][joint].abs().sum() == 0, "fuite : feats non effacées"
assert lab["seq"][lab["seq"] != -100].max() < 4 and lab["loop"][joint].max() < 7
ok(f"{frac:.1%} des bases masquées ; adjacence et feats effacées aux positions masquées")

# ------------------------------------------------------------------ 5
step("Encodeur complet : forward sur 107 et 130 nt")
full = RNAEncoder().to(DEVICE)
n_params = sum(p.numel() for p in full.parameters())
with torch.no_grad():
    h107 = full(b)
    h130 = full(next(iterate_batches([T130], 4, device=DEVICE)))
assert h107.shape == (8, 107, full.d) and h130.shape == (4, 130, full.d)
assert torch.isfinite(h107).all() and torch.isfinite(h130).all()
ok(f"{n_params / 1e6:.2f} M paramètres ; sorties finies pour les deux longueurs")

# ------------------------------------------------------------------ 6
step("Pré-entraînement : sur-apprentissage d'un batch (la loss doit chuter)")
torch.manual_seed(0)
small = MLMModel(RNAEncoder(d=64, h=4, n_layers=2, ff=128)).to(DEVICE)
opt = torch.optim.Adam(small.parameters(), lr=2e-3)
mb, lab = apply_mlm_mask(b)                          # masque fixé
losses = []
for _ in range(80):
    loss = mlm_loss(small(mb), lab)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
assert losses[-1] < 0.7 * losses[0], f"la loss MLM ne baisse pas : {losses[0]:.3f} -> {losses[-1]:.3f}"
ok(f"loss MLM {losses[0]:.3f} -> {losses[-1]:.3f}")

# ------------------------------------------------------------------ 7
step("Sauvegarde / chargement de l'encodeur dans le modèle de régression")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "enc.pt")
    torch.save(small.enc.state_dict(), path)
    enc = RNAEncoder(d=64, h=4, n_layers=2, ff=128)
    enc.load_state_dict(torch.load(path, map_location="cpu"))
same = all(torch.equal(a.cpu(), c) for a, c in zip(small.enc.state_dict().values(), enc.state_dict().values()))
assert same, "poids différents après rechargement"
reg = RegModel(enc).to(DEVICE)
ok("poids identiques après rechargement ; tête de régression branchée")

# ------------------------------------------------------------------ 8
step("Loss torch = métrique du projet")
rng = np.random.default_rng(0)
p_np, y_np = rng.random((4, 107, 5)).astype("float32"), rng.random((4, 68, 5)).astype("float32")
l_torch = mcrmse_loss(torch.from_numpy(p_np), torch.from_numpy(y_np)).item()
l_proj = mcrmse(y_np, p_np[:, :68], cols=list(range(5)))
assert abs(l_torch - l_proj) < 1e-4, f"{l_torch} != {l_proj}"
ok(f"mcrmse_loss torch {l_torch:.5f} = metrics.mcrmse (5 cibles) {l_proj:.5f}")

# ------------------------------------------------------------------ 9
step("Fine-tuning : sur-apprentissage de 32 molécules (la loss doit chuter)")
data = {**X, "y": Y}
opt = torch.optim.Adam(reg.parameters(), lr=2e-3)
first = last = None
for ep in range(15):
    for bb in iterate_batches([data], 8, shuffle=True, rng=rng, device=DEVICE):
        reg.train()
        loss = mcrmse_loss(reg(bb), bb["y"])
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
assert last < first, f"la loss de régression ne baisse pas : {first:.3f} -> {last:.3f}"
ok(f"loss MCRMSE {first:.3f} -> {last:.3f}")

# ------------------------------------------------------------------ 10
step("Prédictions (train + test 107 et 130)")
P = predict(reg, X)[:, : C.SEQ_SCORED]
P107, P130 = predict(reg, T107), predict(reg, T130)
assert P.shape == Y.shape and P107.shape == (8, 107, 5) and P130.shape == (8, 130, 5)
assert np.isfinite(P).all() and np.isfinite(P107).all() and np.isfinite(P130).all()
ok(f"OOF {P.shape}, test {P107.shape} et {P130.shape}, aucune valeur NaN/inf")
ok(f"MCRMSE officielle sur les 32 molécules d'entraînement : {mcrmse(Y, P):.4f}")

# ------------------------------------------------------------------ 11
step("Format de soumission Kaggle")
g107, g130 = load_processed("test_107"), load_processed("test_130")
ids = list(g107["ids"][:8]) + list(g130["ids"][:8])
sub = to_submission(ids, list(P107) + list(P130))
filled = sub.dropna()
assert list(sub.columns) == ["id_seqpos", *C.TARGETS]
assert len(filled) == 8 * 107 + 8 * 130, f"{len(filled)} lignes remplies"
ok(f"{len(filled)} lignes remplies sur {len(sub)} (normal : seulement 16 molécules prédites ici)")

# ------------------------------------------------------------------ 12
step("RNA-FM (optionnel)")
if not (RNAFM_DIR / "train.npy").exists():
    print(f"    ignoré : {RNAFM_DIR / 'train.npy'} absent (lancer extract_rnafm.py pour tester)")
else:
    XR = {k: v[:8] for k, v in load_inputs("train", rnafm=True).items()}
    assert XR["rnafm"].shape == (8, 107, RNAFM_DIM), XR["rnafm"].shape
    assert np.isfinite(XR["rnafm"].astype("float32")).all(), "embeddings RNA-FM non finis"
    enc_r = RNAEncoder(d=64, h=4, n_layers=2, ff=128, rnafm_dim=RNAFM_DIM)
    missing, unexpected = enc_r.load_state_dict(small.enc.state_dict(), strict=False)
    assert not unexpected and all(m.startswith("rnafm_proj.") for m in missing), (missing, unexpected)
    reg_r = RegModel(enc_r).to(DEVICE)
    PR = predict(reg_r, XR)
    assert PR.shape == (8, 107, 5) and np.isfinite(PR).all()
    ok(f"embeddings {XR['rnafm'].shape} alignés ; encodeur MLM rechargé + projection RNA-FM ; prédictions {PR.shape}")

print(f"\nTOUT EST OK ({time.time() - T0:.0f} s). Étape suivante : runs courts avec les vrais scripts.")

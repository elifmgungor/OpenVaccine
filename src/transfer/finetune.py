"""Fine-tuning du Transformer MLM sur les folds communs du projet.

Identique à train.py : données prétraitées, folds (--folds), molécules d'entraînement
(SN_filter = 1 sauf --all-samples), validation sur SN_filter = 1, loss MCRMSE sur les
5 cibles, EarlyStopping(patience, restore_best_weights) sur la val_loss,
ReduceLROnPlateau(0.5, patience // 2), Adam, et sorties dans outputs/<name>/ :
oof.npy (NaN hors folds entraînés), submission.csv, scores.json, history_fold*.csv.

Spécifique au transfer learning : chargement de l'encodeur pré-entraîné, gel de
l'encodeur pendant --freeze-epochs, learning rate séparé pour l'encodeur (--lr-enc).

Option --rnafm : ajoute en entrée les embeddings du modèle de fondation RNA-FM gelé
(extract_rnafm.py). Avec --ckpt none, c'est le transfert depuis RNA-FM seul (tl_rnafm).
La projection RNA-FM est une couche neuve : elle n'est jamais gelée et apprend au
learning rate de la tête.
Loss sur cinq cibles ; checkpoint et arrêt anticipé sélectionnés sur la
validation à cinq cibles. Le score officiel à trois cibles est aussi enregistré.
L'encodeur préentraîné peut être gelé au début et utilise un learning rate séparé.
"""
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config as C
from src.evaluation.metrics import mcrmse
from src.evaluation.submission import to_submission
from src.preprocessing.data import load_processed
from .data import RNAFM_DIM, load_inputs, sn_weight, iterate_batches
from .mlm import seed_all
from .model import RNAEncoder, RegModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALL_IDX = list(range(len(C.TARGETS)))


def mcrmse_loss(pred, y, w=None):
    """Équivalent torch de metrics.mcrmse_loss (5 cibles) ; pondération optionnelle."""
    pred = pred[:, : C.SEQ_SCORED]
    if w is None:
        mse = ((pred - y) ** 2).mean(dim=(0, 1))
    else:
        mse = ((pred - y) ** 2 * w[:, None, None]).mean(1).sum(0) / w.sum().clamp_min(1e-8)
    return torch.sqrt(mse + 1e-8).mean()


@torch.no_grad()
def predict(model, group, bs=256):
    """Prédictions (N, L, 5) dans l'ordre des lignes du groupe."""
    model.eval()
    return np.concatenate([model(b).cpu().numpy() for b in iterate_batches([group], bs, device=DEVICE)])


def take(d, idx):
    return {k: v[idx] for k, v in d.items()}


def set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def run(args):
    pretrained = args.ckpt.lower() != "none"
    use_rnafm = getattr(args, "rnafm", False)
    if use_rnafm:
        default = "tl_mlm_rnafm" if pretrained else "tl_rnafm"
    else:
        default = "tl" if pretrained else "tl_scratch"
    name = args.name or default
    out_dir = C.OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all()
    rng = np.random.default_rng(C.SEED)

    # données prétraitées par scripts/preprocess.py (comme experiments/from_scratch/train.py)
    train = load_processed("train")
    X, Y = load_inputs("train", rnafm=use_rnafm), train["targets"]
    fold = train[f"fold_{args.folds}"]
    clean = train["sn_filter"] == 1
    W = sn_weight(train["signal_to_noise"]) if args.sn_weight else None
    test_groups = [load_processed(f"test_{L}") for L in (107, 130)]
    test_X = [load_inputs(f"test_{L}", rnafm=use_rnafm) for L in (107, 130)]

    folds = [args.only_fold] if args.only_fold is not None else range(C.N_FOLDS)
    oof = np.full_like(Y, np.nan)
    test_preds = [np.zeros((*g["pairs"].shape, len(C.TARGETS))) for g in test_groups]
    fold_scores = {}

    for k in folds:
        print(f"\n===== {name} | fold {k} =====")
        is_val = fold == k
        tr = ~is_val & (True if args.all_samples else clean)
        va = is_val & clean                       # validation toujours sur les molécules fiables
        tr_idx = np.where(tr)[0]
        if args.label_frac < 1:                   # courbes d'apprentissage
            tr_idx = np.sort(rng.choice(tr_idx, int(args.label_frac * len(tr_idx)), replace=False))
        tr_data = {**take(X, tr_idx), "y": Y[tr_idx]}
        if W is not None:
            tr_data["w"] = W[tr_idx]
        va_data = take(X, va)

        enc = RNAEncoder(rnafm_dim=RNAFM_DIM if use_rnafm else 0)
        if pretrained:
            missing, unexpected = enc.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
            if unexpected or any(not m.startswith("rnafm_proj.") for m in missing):
                raise RuntimeError(f"checkpoint incompatible : manquants {missing}, inattendus {unexpected}")
        model = RegModel(enc).to(DEVICE)
<<<<<<< HEAD
        lr_enc = args.lr_enc if pretrained else args.lr   # sans pré-entraînement : protocole train.py
        new = list(model.enc.rnafm_proj.parameters()) if model.enc.rnafm_proj is not None else []
        new_ids = {id(p) for p in new}
        enc_params = [p for p in model.enc.parameters() if id(p) not in new_ids]
        opt = torch.optim.Adam([{"params": enc_params, "lr": lr_enc},
                                {"params": list(model.head.parameters()) + new, "lr": args.lr}])
=======
        lr_enc = args.lr_enc if pretrained else args.lr   # sans pré-entraînement : protocole experiments/from_scratch/train.py
        opt = torch.optim.Adam([{"params": model.enc.parameters(), "lr": lr_enc},
                                {"params": model.head.parameters(), "lr": args.lr}])
>>>>>>> upstream/main
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(   # = keras ReduceLROnPlateau
            opt, factor=0.5, patience=max(1, args.patience // 2), threshold=1e-4, threshold_mode="abs")
        best, best_state, wait, history = float("inf"), None, 0, []

        for ep in range(args.epochs):
            set_trainable(model.enc, not pretrained or ep >= args.freeze_epochs)
            for p in new:                                     # projection RNA-FM : jamais gelée
                p.requires_grad = True
            model.train(); tl, nb = 0.0, 0
            for b in iterate_batches([tr_data], args.batch_size, shuffle=True, rng=rng, device=DEVICE):
                loss = mcrmse_loss(model(b), b["y"], b.get("w"))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tl += loss.item(); nb += 1
            P_va = predict(model, va_data)[:, : C.SEQ_SCORED]
            val_loss = mcrmse(Y[va], P_va, cols=ALL_IDX)       # = val_loss Keras (mcrmse_loss, 5 cibles)
            plateau.step(val_loss)
            history.append({"epoch": ep, "loss": tl / nb, "val_loss": val_loss,
                            "val_mcrmse": mcrmse(Y[va], P_va), "lr": opt.param_groups[1]["lr"]})
            print(f"ep {ep:3d}  loss {tl / nb:.4f}  val_loss {val_loss:.4f}  "
                  f"val_mcrmse {history[-1]['val_mcrmse']:.4f}")
            if val_loss < best:                                # EarlyStopping(restore_best_weights)
                best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
        pd.DataFrame(history).to_csv(out_dir / f"history_fold{k}.csv", index=False)
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / f"model_fold{k}.pt")

        oof[is_val] = predict(model, take(X, is_val))[:, : C.SEQ_SCORED]
        fold_scores[k] = mcrmse(Y[va], oof[va])
        print(f"fold {k} MCRMSE = {fold_scores[k]:.4f}")
        for i, g in enumerate(test_X):
            test_preds[i] += predict(model, g) / len(folds)

    done = np.isin(fold, list(folds)) & clean
    per_target = np.sqrt(np.nanmean((Y[done] - oof[done]) ** 2, axis=(0, 1)))
    scores = {
        "cv_mcrmse": mcrmse(Y[done], oof[done]),
        "cv_mcrmse_5targets": mcrmse(Y[done], oof[done], cols=ALL_IDX),
        "fold_mcrmse": fold_scores,
        "rmse_per_target": dict(zip(C.TARGETS, per_target.round(4).tolist())),
        "args": vars(args),
    }
    print(json.dumps(scores, indent=2))

    np.save(out_dir / "oof.npy", oof)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    ids = [i for g in test_groups for i in g["ids"]]
    preds = [p for arr in test_preds for p in arr]
    to_submission(ids, preds).to_csv(out_dir / "submission.csv", index=False)
    print(f"-> {out_dir}")

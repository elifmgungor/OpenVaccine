"""Figures du rapport (Stratégie 2), générées à partir des fichiers déjà produits.

    python -m experiments.transfer.figures                                   # runs tl, tl_scratch, mean
    python -m experiments.transfer.figures --runs tl tl_scratch gru gnn mean # avec les modèles de l'équipe

Lit :
    outputs/mlm_pretrain/history.csv           (experiments/transfer/preexperiments/from_scratch/train.py)
    outputs/<run>/scores.json                  (experiments/from_scratch/train.py / experiments/transfer/finetune.py)
    results/transfer/loop_errors_<run>.csv    (python -m experiments.transfer.compare --runs ...)
Écrit dans figures/transfer/ (PNG, 200 dpi) :
    fig_pretraining.png      précision sur les bases masquées + perte MLM
    fig_targets.png          RMSE par cible et par modèle
    fig_loop_gain.png        gain relatif à la référence mean, par type de boucle
    fig_loop_diff.png        différence d'erreur tl − tl_scratch (type de boucle × cible)
    fig_learning_curves.png  tl vs tl_scratch à 25 / 50 / 100 % des données (si les runs existent)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src import config as C  # noqa: E402

LOOP_NAMES = {"S": "Tige (S)", "M": "Multi-boucle (M)", "I": "Boucle interne (I)",
              "B": "Renflement (B)", "H": "Épingle (H)", "E": "Extrémité libre (E)",
              "X": "Boucle externe (X)"}
RUN_LABELS = {"tl": "TL (pré-entraîné)", "tl_scratch": "Sans pré-entraînement (tl_scratch)",
              "mean": "Référence mean", "tl_rnafm": "RNA-FM gelé (tl_rnafm)",
              "tl_mlm_rnafm": "MLM + RNA-FM (tl_mlm_rnafm)"}
RUN_COLORS = {"tl": "#2a6fdb", "tl_scratch": "#9aa0a6", "mean": "#d4d4d4",
              "tl_rnafm": "#e07b39", "tl_mlm_rnafm": "#3a9d5d"}
OTHER_COLORS = ["#e07b39", "#3a9d5d", "#8e5cc4", "#c94f6d"]
BLUE, GREY = "#2a6fdb", "#7a7f85"
SCORED = [C.TARGETS[i] for i in C.SCORED_IDX]
FIG_DIR = C.ROOT / "figures" / "transfer"


def fr(v, nd=3):
    """Nombre au format français (virgule décimale)."""
    return f"{v:.{nd}f}".replace(".", ",")


COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def label(run):
    return RUN_LABELS.get(run, run)


def color(run, k):
    return RUN_COLORS.get(run, OTHER_COLORS[k % len(OTHER_COLORS)])


def target_label(t):
    return f"{t} *" if t in SCORED else t


def style():
    plt.rcParams.update({
        "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3, "axes.axisbelow": True,
    })


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def load_scores(run):
    p = C.OUTPUT_DIR / run / "scores.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_loops(run):
    p = C.ROOT / "results" / "transfer" / f"loop_errors_{run}.csv"
    return pd.read_csv(p).set_index("loop") if p.exists() else None


# ------------------------------------------------------------------ pré-entraînement
def fig_pretraining():
    path = C.OUTPUT_DIR / "mlm_pretrain" / "history.csv"
    if not path.exists():
        print(f"  (ignoré) {path} absent")
        return
    h = pd.read_csv(path)
    acc = h.masked_nt_acc * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    a1.plot(h.epoch, acc, color=BLUE, lw=2)
    a1.axhline(25, ls="--", lw=1, color=GREY)
    a1.text(h.epoch.max(), 28, "hasard (1 base sur 4)", ha="right", color=GREY, fontsize=9)
    a1.annotate(f"{fr(acc.iloc[0], 0)} %", (h.epoch.iloc[0], acc.iloc[0]),
                xytext=(6, -14), textcoords="offset points", fontweight="bold")
    a1.annotate(f"{fr(acc.iloc[-1], 0)} %", (h.epoch.iloc[-1], acc.iloc[-1]),
                xytext=(-10, 8), textcoords="offset points", ha="right", fontweight="bold")
    a1.set(ylim=(0, 100), xlabel="Epoch", ylabel="Précision (%)",
           title="Précision sur les nucléotides masqués (validation)")

    a2.plot(h.epoch, h.loss, color=GREY, lw=1.5, label="Entraînement")
    a2.plot(h.epoch, h.val_loss, color=BLUE, lw=2, label="Validation")
    best = h.val_loss.idxmin()
    a2.scatter(h.epoch[best], h.val_loss[best], color=BLUE, zorder=3)
    a2.annotate(f"meilleure validation : {fr(h.val_loss[best])} (epoch {int(h.epoch[best])})",
                (h.epoch[best], h.val_loss[best]), xytext=(-10, 40), textcoords="offset points",
                ha="right", fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
    a2.yaxis.set_major_formatter(COMMA)
    a2.set(ylim=(0, None), xlabel="Epoch", ylabel="Entropie croisée", title="Perte MLM")
    a2.legend(frameon=False)

    fig.suptitle(f"Pré-entraînement MLM : de {fr(acc.iloc[0], 0)} % à {fr(acc.iloc[-1], 0)} % "
                 "de nucléotides reconstruits", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig_pretraining.png")


# ------------------------------------------------------------------ par cible
def fig_targets(runs):
    data = {r: s["rmse_per_target"] for r in runs if (s := load_scores(r))}
    if not data:
        print("  (ignoré) aucun scores.json trouvé")
        return
    x = np.arange(len(C.TARGETS))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    top = 0
    for k, (r, d) in enumerate(data.items()):
        vals = [d[t] for t in C.TARGETS]
        top = max(top, max(vals))
        bars = ax.bar(x - 0.4 + w * (k + 0.5), vals, w, label=label(r), color=color(r, k),
                      edgecolor="white")
        ax.bar_label(bars, labels=[fr(v) for v in vals], fontsize=7, padding=2, rotation=90)
    ax.set_xticks(x, [target_label(t) for t in C.TARGETS])
    ax.yaxis.set_major_formatter(COMMA)
    ax.set(ylim=(0, top * 1.2), ylabel="RMSE (validation croisée, plus bas = meilleur)",
           title="Erreur par cible")
    ax.legend(frameon=False, ncol=len(data), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.text(1, -0.24, "* cibles évaluées par la MCRMSE officielle", transform=ax.transAxes,
            ha="right", fontsize=8, color=GREY)
    save(fig, "fig_targets.png")


# ------------------------------------------------------------------ par type de boucle
def fig_loop_gain(runs):
    base = load_loops("mean")
    if base is None:
        print("  (ignoré) loop_errors_mean.csv absent : lancer compare avec le run mean")
        return
    models = [r for r in runs if r != "mean" and load_loops(r) is not None]
    if not models:
        print("  (ignoré) aucun loop_errors_<run>.csv pour les modèles demandés")
        return
    loops = [lp for lp in C.LOOP_VOCAB if lp in base.index]
    x = np.arange(len(loops))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    top = 0
    for k, r in enumerate(models):
        df = load_loops(r)
        gain = (1 - df.loc[loops, SCORED] / base.loc[loops, SCORED]).mean(axis=1) * 100
        top = max(top, gain.max())
        bars = ax.bar(x - 0.4 + w * (k + 0.5), gain, w, label=label(r), color=color(r, k),
                      edgecolor="white")
        ax.bar_label(bars, labels=[fr(g, 1) for g in gain], fontsize=8, padding=2)
    ticks = [f"{LOOP_NAMES.get(lp, lp)}\n{int(base.loc[lp, 'n_pos']):,} pos.".replace(",", " ")
             for lp in loops]
    ax.set_xticks(x, ticks, fontsize=8)
    ax.set(ylim=(0, top * 1.15), ylabel="Réduction d'erreur par rapport à mean (%)",
           title="Gain relatif par type de boucle (moyenne des 3 cibles évaluées)")
    ax.legend(frameon=False, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig_loop_gain.png")


def fig_loop_diff(a="tl", b="tl_scratch"):
    da, db = load_loops(a), load_loops(b)
    if da is None or db is None:
        print(f"  (ignoré) loop_errors_{a}.csv ou loop_errors_{b}.csv absent")
        return
    loops = [lp for lp in C.LOOP_VOCAB if lp in da.index and lp in db.index]
    D = (da.loc[loops, C.TARGETS] - db.loc[loops, C.TARGETS]).to_numpy()
    lim = np.abs(D).max()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            v = D[i, j]
            txt = "0,0000" if round(v, 4) == 0 else ("+" if v > 0 else "−") + fr(abs(v), 4)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_xticks(range(len(C.TARGETS)), [target_label(t) for t in C.TARGETS])
    ax.set_yticks(range(len(loops)), [LOOP_NAMES.get(lp, lp) for lp in loops])
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f"Δ RMSE  (bleu = {a} meilleur, rouge = {b} meilleur)")
    cb.ax.yaxis.set_major_formatter(COMMA)
    ax.set_title(f"Différence d'erreur {a} − {b} par type de boucle et par cible")
    save(fig, "fig_loop_diff.png")


# ------------------------------------------------------------------ courbes d'apprentissage
def fig_learning_curves():
    fracs = [(25, "_25"), (50, "_50"), (100, "")]
    series = {}
    for base in ("tl", "tl_scratch"):
        vals = [load_scores(base + suf) for _, suf in fracs]
        if any(v is None for v in vals):
            print(f"  (ignoré) runs {base}_25 / {base}_50 / {base} incomplets")
            return
        series[base] = [v["cv_mcrmse"] for v in vals]
    x = np.arange(len(fracs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (r, vals) in enumerate(series.items()):
        ax.plot(x, vals, marker="o", lw=2, color=color(r, k), label=label(r))
    for i in x:
        a, b = series["tl"][i], series["tl_scratch"][i]
        for v, above in ((a, a > b), (b, b >= a)):
            ax.annotate(fr(v, 4), (i, v), xytext=(0, 9 if above else -15),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.text(i, -0.16, f"écart tl − scratch : {'+' if a > b else '−'}{fr(abs(a - b), 4)}",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=GREY)
    ax.set_xticks(x, [f"{p} %" for p, _ in fracs])
    ax.set_xlim(-0.3, len(fracs) - 0.7)
    ax.yaxis.set_major_formatter(COMMA)
    ax.set(xlabel="\n\nFraction des molécules d'entraînement",
           ylabel="MCRMSE officielle (plus bas = meilleur)", title="Courbes d'apprentissage")
    ax.legend(frameon=False)
    save(fig, "fig_learning_curves.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["tl", "tl_scratch", "mean"],
                   help="dossiers dans outputs/ (mettre mean en dernier)")
    args = p.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    style()
    print(f"Figures -> {FIG_DIR}")
    fig_pretraining()
    fig_targets(args.runs)
    fig_loop_gain(args.runs)
    fig_loop_diff()
    fig_learning_curves()


if __name__ == "__main__":
    main()

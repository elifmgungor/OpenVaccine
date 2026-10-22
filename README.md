# OpenVaccine : prédiction de la dégradation de l’ARN

Projet du cours IA2 (Master 2 Bioinformatique, Université Paris Cité), construit sur les données de la compétition Kaggle [Stanford OpenVaccine](https://www.kaggle.com/competitions/stanford-covid-vaccine).

L’objectif est de prédire, pour chaque nucléotide d’une molécule d’ARN, sa réactivité et sa dégradation dans plusieurs conditions expérimentales. Le projet étudie trois questions, chacune dans un volet séparé :

1. **Modèles from scratch** : quelle architecture (récurrente, attention, graphe) et quel niveau d’information structurale donnent les meilleures prédictions ?
2. **Transfer learning** : un préentraînement auto-supervisé (MLM) ou les embeddings d’un modèle de fondation (RNA-FM) améliorent-ils les prédictions ?
3. **Adaptation d’une solution Kaggle** : comment se situe une architecture communautaire performante sur notre protocole commun ?

Les trois volets partagent le même prétraitement, les mêmes folds et la même métrique, ce qui rend leurs scores comparables.

## Sommaire

- [Résultats en bref](#résultats-en-bref)
- [Installation](#installation)
- [Données et prétraitement](#données-et-prétraitement)
- [Volet 1 : modèles from scratch](#volet-1--modèles-from-scratch)
- [Volet 2 : transfer learning (MLM et RNA-FM)](#volet-2--transfer-learning-mlm-et-rna-fm)
- [Volet 3 : adaptation d’une solution Kaggle](#volet-3--adaptation-dune-solution-kaggle)
- [Ensemble de modèles](#ensemble-de-modèles)
- [Évaluation](#évaluation)
- [Tests](#tests)
- [Organisation du dépôt](#organisation-du-dépôt)
- [Limites](#limites)

## Résultats en bref

MCRMSE officielle sur les cinq folds communs (plus bas est meilleur). Les valeurs « ± » sont l’écart-type entre folds.

| Volet | Modèle | Entrées | MCRMSE |
| --- | --- | --- | --- |
| Référence | Moyenne par position (`mean`) | – | 0,4256 ± 0,0042 |
| From scratch | BiGRU | `seq` | ≈ 0,309 (OOF) |
| From scratch | **BiGRU** | `annot` | **0,2493 ± 0,0055** |
| From scratch | Conv1D + BiGRU | `annot` | 0,2511 ± 0,0055 |
| From scratch | BiGRU + Attention | `annot` | 0,2579 ± 0,0048 |
| From scratch | Transformer | `annot` | 0,3041 ± 0,0062 |
| From scratch | GCN | `annot` | 0,3482 ± 0,0049 |
| From scratch | GAT | `annot` | 0,3524 ± 0,0047 |
| Transfert | Transformer sans préentraînement (`tl_scratch`) | `annot` + biais structuraux | 0,2543 ± 0,0056 |
| Transfert | Transformer préentraîné MLM (`tl`) | `annot` + biais structuraux | 0,2561 ± 0,0056 |
| Transfert | + embeddings RNA-FM gelés (`tl_mlm_rnafm`) | idem + RNA-FM | 0,2374 ± 0,0047 |
| Transfert | **RNA-FM gelé, sans MLM (`tl_rnafm`)** | idem + RNA-FM | **0,2356 ± 0,0049** |
| Kaggle | **AE + GNN + Attention + CNN** | 39 caractéristiques + 5 adjacences | **0,2266 ± 0,0054** |

Ce qu’il faut retenir :

- Les **annotations structurales** (structure secondaire, type de boucle, statistiques BPPS) réduisent l’erreur du BiGRU d’environ 19 % par rapport à la séquence seule. Ajouter les **interactions explicites** entre bases (niveau `graph`) n’apporte rien aux modèles testés.
- Les **modèles récurrents** dominent nettement les modèles sur graphe dans notre configuration.
- Le **préentraînement MLM maison** n’améliore pas le score, même avec 25 % ou 50 % des données étiquetées. En revanche, les **embeddings du modèle de fondation RNA-FM** font gagner environ 0,02 de MCRMSE et donnent le meilleur résultat parmi nos propres pipelines.
- L’**adaptation de la solution Kaggle** obtient le meilleur score global, avec des caractéristiques plus riches et un préentraînement par autoencodeur.

Attention : ces scores comparent des pipelines complets (caractéristiques, budgets et nombres de paramètres différents), pas uniquement des architectures. Voir [Limites](#limites).

## Installation

Python 3.10 ou plus récent est recommandé.

```bash
git clone https://github.com/elifmgungor/OpenVaccine.git
cd OpenVaccine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # volets 1 et 3 (TensorFlow / Keras 3)
pip install -r requirements-transfer.txt   # volet 2 (PyTorch), inclut requirements.txt
```

Pour les expériences avec RNA-FM, installer aussi :

```bash
pip install rna-fm
```

Un GPU est fortement conseillé pour le volet 2. Le notebook [`notebooks/Run_transfert.ipynb`](notebooks/Run_transfert.ipynb) permet de tout exécuter sur Google Colab (données et sorties stockées sur Google Drive).

## Données et prétraitement

Les données ne sont pas versionnées. Télécharger depuis la [page de la compétition](https://www.kaggle.com/competitions/stanford-covid-vaccine/data) et placer dans `data/` :

```text
data/
  train.json            (ou train.json.zip)
  test.json             (ou test.json.zip)
  sample_submission.csv
  bpps/                 matrices de probabilités d’appariement, une par molécule (.npy)
```

Puis lancer le prétraitement commun :

```bash
python -m scripts.preprocess
```

Il produit `data/processed/train.npz`, `test_107.npz` et `test_130.npz`. Pour chaque nucléotide, il encode la base, la structure secondaire et le type de boucle, et calcule quatre caractéristiques : somme, maximum et entropie des BPPS, et distance au partenaire d’appariement. Les cinq cibles et les affectations de folds de [`folds/`](folds/) sont conservées.

Les folds sont **fixes et partagés** par tous les volets (`folds/folds_kfold.csv` par défaut, `folds/folds_group.csv` en alternative). Seules les molécules fiables (`SN_filter = 1`, soit 1 589 ARN) servent à l’entraînement et à l’évaluation, sauf option `--all-samples`.

## Volet 1 : modèles from scratch

Code : [`src/models/`](src/models/) (Keras 3). Script principal : [`experiments/from_scratch/train.py`](experiments/from_scratch/train.py).

### Modèles disponibles

| Nom (`--model`) | Architecture |
| --- | --- |
| `mean` | Référence : moyenne des cibles par position |
| `gru` | BiGRU (baseline) |
| `conv_gru` | Conv1D puis BiGRU |
| `attention_gru` | BiGRU suivi d’une attention multi-têtes |
| `transformer` | Transformer encodeur, avec biais BPPS au niveau `graph` |
| `gcn` | Graph Convolutional Network (option `relational` pour un R-GCN) |
| `gat` | Graph Attention Network |
| `mlp` | Référence auxiliaire : chaque base traitée isolément |

### Niveaux d’information structurale

L’option `--level` contrôle ce que voit le modèle :

- `seq` : séquence de nucléotides seule ;
- `annot` : séquence + annotations par base (structure, type de boucle, caractéristiques BPPS) ;
- `graph` : `annot` + interactions entre bases (matrice BPPS et paires), utilisées par l’attention du Transformer ou comme arêtes du GCN et du GAT.

### Entraîner un modèle

```bash
# BiGRU avec annotations, validation croisée sur les 5 folds
python -m experiments.from_scratch.train --model gru --level annot --name cv_gru_annot

# Transformer avec interactions explicites et hyperparamètres personnalisés
python -m experiments.from_scratch.train --model transformer --level graph \
    --params '{"d_model": 64, "num_heads": 4, "n_layers": 2, "ff_dim": 128, "dropout": 0.2}' \
    --name cv_transformer_graph

# Test rapide sur un seul fold
python -m experiments.from_scratch.train --model gru --level annot --only-fold 0 --epochs 2 --name gru_pilot

# Référence moyenne (nécessaire pour les comparaisons du volet 2)
python -m experiments.from_scratch.train --model mean --name mean
```

Principales options : `--folds {kfold,group}`, `--only-fold K`, `--optimizer {adamw,adam}`, `--lr` (0,001), `--batch-size` (32), `--epochs` (40), `--patience` (8), `--seed` (42), `--skip-test` (pas d’inférence sur le test Kaggle), `--all-samples` (inclure les molécules bruitées).

Chaque run écrit dans `outputs/<name>/` : `config.json`, `scores.json`, les poids `fold*.weights.h5`, les historiques `history_fold*.csv`, les prédictions OOF `oof.npy` et, sauf `--skip-test`, un `submission.csv` au format Kaggle.

### Reproduire la comparaison complète

Le script `run.py` lance les dix configurations du rapport (six architectures en `annot`, plus BiGRU en `seq` et Transformer, GCN et GAT en `graph`) avec le protocole commun. Les runs déjà terminés sont réutilisés.

```bash
python -m experiments.from_scratch.run --jobs 2             # les 10 configurations
python -m experiments.from_scratch.run --select gru_annot gat_graph   # un sous-ensemble
python -m experiments.from_scratch.verify                   # recalcule les scores depuis les OOF
python -m experiments.from_scratch.figures                  # régénère les figures
```

Protocole : cinq folds fixes, seed 42, AdamW (lr 0,001, weight decay 1e-4), lots de 32, au plus 40 époques, arrêt anticipé après huit époques sans amélioration de la MCRMSE officielle de validation. Le test n’intervient jamais dans l’apprentissage.

### Résultats

![Comparaison des six architectures avec les entrées annot](figures/from_scratch/cv_architectures.png)

![Apport des informations structurales selon le niveau d’entrée](figures/from_scratch/cv_structure.png)

L’erreur par cible est présentée dans [`figures/from_scratch/cv_targets.png`](figures/from_scratch/cv_targets.png).

## Volet 2 : transfer learning (MLM et RNA-FM)

Code : [`src/transfer/`](src/transfer/) (PyTorch). Scripts : [`experiments/transfer/`](experiments/transfer/).

Le modèle est un Transformer avec biais structuraux. Trois sources d’initialisation ou d’information sont comparées :

- **MLM** : l’encodeur est d’abord préentraîné à reconstruire des nucléotides masqués (sur les séquences de train et de test, sans leurs labels), puis affiné sur les cinq cibles ;
- **sans préentraînement** (`tl_scratch`) : même architecture initialisée aléatoirement, comme contrôle ;
- **RNA-FM** : les embeddings du modèle de fondation [RNA-FM](https://github.com/ml4bio/RNA-FM) (dernière couche, 640 dimensions par nucléotide, modèle gelé) sont ajoutés en entrée, avec ou sans préentraînement MLM.

### 1. Préentraînement MLM

```bash
python -m experiments.transfer.pretrain --epochs 100
```

Options : `--mask-p` (proportion de bases masquées, 0,15), `--joint-p` (probabilité de masquer aussi structure et boucle, 0,5), `--span` (longueur des spans masqués), `--lr`, `--batch-size`, `--out`. Sortie : `outputs/mlm_pretrain/encoder.pt` et `history.csv`.

### 2. Fine-tuning

```bash
python -m experiments.transfer.finetune                                  # avec MLM        -> outputs/tl/
python -m experiments.transfer.finetune --ckpt none                      # sans MLM        -> outputs/tl_scratch/
python -m experiments.transfer.finetune --label-frac 0.25 --name tl_25   # 25 % des données étiquetées
python -m experiments.transfer.finetune --ckpt none --label-frac 0.25 --name tl_scratch_25
```

Options : `--ckpt` (chemin de l’encodeur ou `none`), `--lr` (tête, 1e-3), `--lr-enc` (encodeur, 1e-4), `--freeze-epochs` (époques avec encodeur gelé, 3), `--label-frac`, `--sn-weight` (pondérer la perte par le rapport signal/bruit), `--folds`, `--only-fold`, `--epochs` (100), `--patience` (10), `--batch-size` (64).

Sorties dans `outputs/<name>/` : `oof.npy`, `submission.csv`, `scores.json`, `history_fold*.csv`, `model_fold*.pt`.

### 3. Avec les embeddings RNA-FM

Extraire une fois les embeddings (environ 0,9 Go en float16). Les poids de RNA-FM sont téléchargés depuis [Hugging Face](https://huggingface.co/cuhkaih/rnafm) ; le notebook Colab montre la procédure.

```bash
python extract_rnafm.py            # -> outputs/rnafm/{train,test_107,test_130}.npy
python -m experiments.transfer.finetune --rnafm --ckpt none --name tl_rnafm      # RNA-FM seul
python -m experiments.transfer.finetune --rnafm --name tl_mlm_rnafm              # MLM + RNA-FM
```

### 4. Comparaison et figures

```bash
python -m experiments.transfer.compare --runs tl tl_scratch tl_rnafm tl_mlm_rnafm mean
python -m experiments.transfer.compare --runs tl_25 tl_scratch_25
python -m experiments.transfer.figures
```

La comparaison avec `mean` nécessite d’avoir lancé la référence du volet 1 (`--model mean --name mean`). Les tableaux publiés sont dans [`results/transfer/`](results/transfer/) : `main_table.csv` (scores globaux et par cible), `bootstrap.csv` et les erreurs par type de boucle `loop_errors_*.csv`.

### Résultats

![Effet du préentraînement selon la quantité de données étiquetées](figures/transfer/fig_learning_curves.png)

Le préentraînement MLM n’apporte pas de gain mesurable : les écarts avec `tl_scratch` restent inférieurs à 0,002 de MCRMSE quelle que soit la fraction de données étiquetées. Les embeddings RNA-FM améliorent en revanche toutes les cibles (MCRMSE 0,2357 contre 0,2544 sans).

Figures complémentaires : [erreur par cible](figures/transfer/fig_targets.png), [suivi du préentraînement](figures/transfer/fig_pretraining.png), [erreur par type de boucle](figures/transfer/fig_loop_diff.png) et [gain par type de boucle](figures/transfer/fig_loop_gain.png).

## Volet 3 : adaptation d’une solution Kaggle

Code : [`src/kaggle/model.py`](src/kaggle/model.py) (Keras 3). Script : [`experiments/kaggle/train.py`](experiments/kaggle/train.py).

Nous adaptons le notebook communautaire de [mrkmakr, « AE pretrain + GNN + Attn + CNN »](https://www.kaggle.com/code/mrkmakr/covid-ae-pretrain-gnn-attn-cnn) : autoencodeur débruiteur préentraîné, convolutions à plusieurs tailles de noyau, agrégation sur graphe et attention à quatre têtes. Ce n’est pas l’ensemble final du gagnant de la compétition.

Pour respecter le protocole commun, l’autoencodeur est réentraîné pour chaque fold, uniquement sur les entrées de son ensemble d’entraînement, et le modèle utilise les mêmes molécules et les mêmes folds que les autres volets. Toutes les différences avec le notebook d’origine sont détaillées dans [`experiments/kaggle/README.md`](experiments/kaggle/README.md).

```bash
python -m experiments.kaggle.train     # 5 folds -> outputs/cv_kaggle_ae_common/
python -m experiments.kaggle.verify    # vérifie les scores et produit figures/kaggle/cv_comparison.png

# Test fonctionnel sur huit ARN (aucune valeur scientifique)
python -m experiments.kaggle.train --smoke --ae-epochs 1 --epochs 1 --only-fold 0 --name smoke_kaggle_ae
```

Options : `--ae-epochs` (préentraînement de l’autoencodeur par fold, 20), `--epochs` (40), `--batch-size` (32), `--only-fold`, `--name`. Une exécution interrompue peut être relancée avec la même commande : les folds terminés sont réutilisés.

### Résultats

![Adaptation Kaggle et baseline BiGRU sur les mêmes cinq folds](figures/kaggle/cv_comparison.png)

MCRMSE moyenne **0,22657 ± 0,00539** sur les cinq folds, et **0,22663** sur les prédictions OOF regroupées.

## Ensemble de modèles

[`scripts/ensemble.py`](scripts/ensemble.py) combine plusieurs runs par moyenne pondérée. Les poids sont optimisés sur les prédictions OOF, jamais sur le test.

```bash
python -m scripts.ensemble --runs cv_gru_annot cv_conv_gru_annot tl_rnafm --name ensemble
```

Chaque nom passé à `--runs` doit correspondre à un dossier de `outputs/` contenant un `oof.npy`.

## Évaluation

Chaque modèle prédit cinq cibles (`reactivity`, `deg_Mg_pH10`, `deg_pH10`, `deg_Mg_50C`, `deg_50C`) sur les 68 premières positions, seules mesurées dans le train. Le score officiel est la **MCRMSE** (moyenne des RMSE par colonne) sur trois cibles : `reactivity`, `deg_Mg_pH10` et `deg_Mg_50C`. L’implémentation se trouve dans [`src/evaluation/metrics.py`](src/evaluation/metrics.py).

Deux façons de résumer la validation croisée sont utilisées : la moyenne ± écart-type des scores des cinq folds, et le score OOF calculé sur l’ensemble des prédictions hors fold regroupées.

## Tests

```bash
python -m unittest discover -s tests -v   # modèles Keras
python -m tests.smoke_transfer            # pipeline de transfert (PyTorch)
```

## Organisation du dépôt

```text
src/
  config.py            Chemins, cibles, vocabulaires, seed
  preprocessing/       Encodage, caractéristiques par base, folds
  evaluation/          Métrique MCRMSE et format de soumission
  training.py          Utilitaires d’entraînement Keras
  models/              Modèles from scratch (Keras)
  transfer/            Transformer, MLM et fine-tuning (PyTorch)
  kaggle/              Architecture Kaggle adaptée (Keras)
experiments/
  from_scratch/        Entraînement, comparaison, vérification, figures
  transfer/            Préentraînement, fine-tuning, comparaison, figures
  kaggle/              Entraînement et vérification
scripts/               Prétraitement commun et ensemble
extract_rnafm.py       Extraction des embeddings RNA-FM
notebooks/             Analyse exploratoire et exécution sur Colab
folds/                 Partitions fixes communes aux trois volets
figures/               Figures du rapport, par volet
results/transfer/      Tableaux de résultats publiés du volet 2
tests/                 Tests unitaires et smoke tests
data/                  Données Kaggle (non versionnées)
outputs/               Checkpoints, historiques et prédictions (non versionnés)
```

## Limites

- Les nombres de paramètres, caractéristiques d’entrée et budgets d’entraînement diffèrent entre volets : les scores comparent des pipelines, pas seulement des architectures.
- Le MLM utilise les séquences de test (sans labels) pour le préentraînement, contrairement à l’autoencodeur du volet Kaggle, entraîné fold par fold.
- Le fine-tuning du volet 2 sélectionne le checkpoint sur les cinq cibles, contre les trois cibles officielles pour les volets 1 et 3.
- RNA-FM est un modèle de fondation externe préentraîné sur un très grand corpus d’ARN : son gain ne se compare pas directement à celui du MLM entraîné sur les seules données de la compétition.
- Une seule seed est étudiée, et le découpage KFold ne garantit pas que des séquences similaires se trouvent dans le même fold.

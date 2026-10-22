"""Encodeur Transformer pré-entraînable (Stratégie 2, PyTorch).

Rangé dans src/transfer/ et PAS dans src/models/ : src/models/ est le registre
des modèles Keras utilisés par experiments/from_scratch/train.py (MODELS), ce modèle-ci est en PyTorch.

Mêmes entrées que les autres modèles : tokens (vocabulaire global à offsets),
feats, bpps, pairs. Biais d'attention 2D (BPP + adjacence) ; tête MLM pour le
pré-entraînement, tête de régression pour le fine-tuning.

Option RNA-FM (rnafm_dim > 0) : les embeddings d'un modèle de fondation gelé
(640 dimensions par nucléotide) sont projetés en d dimensions et ajoutés à
l'embedding, comme les feats. Sans cette option, le modèle est inchangé et les
anciens checkpoints restent compatibles.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C
from src.preprocessing.features import FEATURE_NAMES

OFFSETS = (0, len(C.SEQ_VOCAB), len(C.SEQ_VOCAB) + len(C.STRUCT_VOCAB))
N_CLASSES = (len(C.SEQ_VOCAB), len(C.STRUCT_VOCAB), len(C.LOOP_VOCAB))
MASK_IDS = (C.VOCAB_SIZE, C.VOCAB_SIZE + 1, C.VOCAB_SIZE + 2)   # un [MASK] par canal
CHANNELS = ("seq", "struct", "loop")


def pairs_to_adj(pairs):
    """(B, L) index du partenaire (-1 sinon) -> (B, L, L) adjacence."""
    B, L = pairs.shape
    adj = torch.zeros(B, L, L, device=pairs.device)
    bi, i = (pairs >= 0).nonzero(as_tuple=True)
    adj[bi, i, pairs[bi, i]] = 1.0
    return adj


class GraphBiasedAttentionLayer(nn.Module):
    """Pre-LN Transformer ; scores + a_h·BPP + b_h·adjacence (un poids appris par tête)."""
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.h, self.dk, self.p = h, d // h, drop
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.a_bpp = nn.Parameter(torch.ones(h))
        self.a_adj = nn.Parameter(torch.ones(h))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bpp, adj):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        bias = self.a_bpp.view(1, -1, 1, 1) * bpp[:, None] + self.a_adj.view(1, -1, 1, 1) * adj[:, None]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


class RNAEncoder(nn.Module):
    def __init__(self, d=192, h=6, n_layers=4, ff=768, drop=0.1, max_len=256, rnafm_dim=0):
        super().__init__()
        self.d = d
        self.rnafm_proj = (nn.Sequential(nn.Dropout(drop), nn.Linear(rnafm_dim, d), nn.LayerNorm(d))
                           if rnafm_dim else None)
        self.emb = nn.Embedding(C.VOCAB_SIZE + len(MASK_IDS), d)
        self.feat_proj = nn.Sequential(nn.Linear(len(FEATURE_NAMES), d), nn.LayerNorm(d))
        self.pos = nn.Embedding(max_len, d)
        self.conv = nn.Conv1d(d, d, kernel_size=5, padding=2)    # motifs locaux
        self.layers = nn.ModuleList([GraphBiasedAttentionLayer(d, h, ff, drop) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        tok = b["tokens"].long()
        L = tok.shape[1]
        adj = b["adj"] if "adj" in b else pairs_to_adj(b["pairs"].long())
        x = (self.emb(tok).sum(2) + self.feat_proj(b["feats"])
             + self.pos(torch.arange(L, device=tok.device))[None])
        if self.rnafm_proj is not None:
            if "rnafm" not in b:
                raise KeyError("encodeur RNA-FM : la clé 'rnafm' manque dans le batch (load_inputs(..., rnafm=True))")
            x = x + self.rnafm_proj(b["rnafm"].float())
        x = x + F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, b["bpps"], adj)
        return self.norm(x)


class MLMModel(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.heads = nn.ModuleDict({c: nn.Linear(enc.d, n) for c, n in zip(CHANNELS, N_CLASSES)})

    def forward(self, b):
        h = self.enc(b)
        return {c: head(h) for c, head in self.heads.items()}


class RegModel(nn.Module):
    def __init__(self, enc, drop=0.1):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(enc.d, enc.d), nn.GELU(),
                                  nn.Linear(enc.d, len(C.TARGETS)))

    def forward(self, b):
        return self.head(self.enc(b))

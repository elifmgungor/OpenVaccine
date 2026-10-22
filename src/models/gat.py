"""Modèle 4 : GAT, attention sur le graphe de l'ARN (cours GNN).

    alpha_uv = softmax_u( LeakyReLU( a^T [W x_v || W x_u] ) )   pour u voisin de v
    x'_v     = elu( sum_u alpha_uv W x_u )                      (têtes concaténées)

Contrairement au GCN (poids fixés par la normalisation des degrés), le GAT
apprend l'importance de chaque voisin. Voisins : squelette (+ paires de base
et arêtes BPPS > bpps_min au niveau "graph"), plus la base elle-même.
"""
import keras
from keras import layers, ops

from .common import check_level, embed, model_inputs, output_head, used_inputs
from .graph import NeighbourMask, graph_adjacency


class GATLayer(layers.Layer):
    def __init__(self, units, num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.h, self.dh = num_heads, units // num_heads
        self.W = layers.Dense(units, use_bias=False)
        self.drop = layers.Dropout(dropout)

    def build(self, input_shape):
        self.a_src = self.add_weight(shape=(self.h, self.dh), initializer="glorot_uniform", name="a_src")
        self.a_dst = self.add_weight(shape=(self.h, self.dh), initializer="glorot_uniform", name="a_dst")

    def call(self, x, mask, training=None):
        B, L = ops.shape(x)[0], ops.shape(x)[1]
        wx = ops.reshape(self.W(x), (B, L, self.h, self.dh))                  # (B, L, h, dh)
        e_dst = ops.einsum("blhd,hd->bhl", wx, self.a_dst)                    # cible v
        e_src = ops.einsum("blhd,hd->bhl", wx, self.a_src)                    # source u
        e = ops.leaky_relu(e_dst[:, :, :, None] + e_src[:, :, None, :], negative_slope=0.2)  # (B, h, v, u)
        e = ops.where(mask[:, None] > 0, e, -1e9)                             # seulement les voisins
        alpha = self.drop(ops.softmax(e, axis=-1), training=training)
        out = ops.einsum("bhvu,buhd->bvhd", alpha, wx)
        return ops.elu(ops.reshape(out, (B, L, self.h * self.dh)))


def build_model(n_feats, level="graph", embed_dim=32, units=128, num_heads=4, n_layers=4,
                dropout=0.2, bpps_min=0.1):
    check_level(level)
    inputs = model_inputs(n_feats)
    x = layers.Dense(units)(embed(inputs, embed_dim, level))

    A = graph_adjacency(x, inputs, level)                                     # (B, R, L, L)
    if level == "graph":   # arêtes BPPS : seulement les appariements assez probables
        A = ops.concatenate([A[:, :2], ops.cast(A[:, 2:] > bpps_min, "float32")], axis=1)
    mask = NeighbourMask()(A)                                                 # (B, L, L) + self-loops

    for _ in range(n_layers):
        h = GATLayer(units, num_heads, dropout)(x, mask)
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(h))

    return keras.Model(used_inputs(inputs, level, ("bpps", "pairs")), output_head(x), name=f"gat_{level}")

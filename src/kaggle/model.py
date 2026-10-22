"""Kaggle community AE + GNN + attention + CNN, adapted for Keras 3.

Source: https://www.kaggle.com/code/mrkmakr/covid-ae-pretrain-gnn-attn-cnn
The runner controls pretraining data, folds, seeds and official scoring.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from src.evaluation.metrics import mcrmse_loss


def attention(x_inner, x_outer, n_factor):
    x_Q =  L.Conv1D(n_factor, 1, activation='linear',
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_inner)
    x_K =  L.Conv1D(n_factor, 1, activation='linear',
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_V =  L.Conv1D(n_factor, 1, activation='linear',
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: tf.keras.ops.matmul(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
    att = L.Lambda(lambda c: tf.keras.ops.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: tf.keras.ops.matmul(c[0], c[1]))([att, x_V])
    return att

def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor,
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

def res(x, unit, kernel = 3, rate = 0.1):
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])

def forward(x, unit, kernel = 3, rate = 0.1):
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    h = L.Dropout(rate)(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate)
    return h

def adj_attn(x, adj, unit):
    x_a = x
    x_as = []
    for _ in range(2):
        x_a = forward(x_a, unit)
        x_a = tf.keras.ops.matmul(adj, x_a)  # Agréger les voisins de chaque nucléotide.
        x_as.append(x_a)
    x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a


def get_base():
    """Encodeur partagé : convolutions, agrégation du graphe et attention."""
    node = tf.keras.Input(shape = (None, 39), name = "node")
    adj = tf.keras.Input(shape = (None, None, 5), name = "adj")

    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis = 3)([adj, adj_learned])

    xs = [node]
    x1 = forward(node, 128, kernel = 3, rate = 0.0)
    x2 = forward(x1, 64, kernel = 6, rate = 0.0)
    x3 = forward(x2, 32, kernel = 15, rate = 0.0)
    x4 = forward(x3, 16, kernel = 30, rate = 0.0)
    x = L.Concatenate()([x1, x2, x3, x4])

    for unit in [64, 32]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel = 30)

        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)

    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs = [node, adj], outputs = x)
    return model


def get_ae_model(base):
    """Reconstruire les 39 caractéristiques à partir des entrées bruitées."""
    node = tf.keras.Input(shape = (None, 39), name = "node")
    adj = tf.keras.Input(shape = (None, None, 5), name = "adj")

    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64, rate = 0.3)
    p = L.Dense(39, "sigmoid")(x)

    loss = L.Lambda(
        lambda z: -tf.keras.ops.mean(
            20 * z[0] * tf.keras.ops.log(z[1] + 1e-4)
            + (1 - z[0]) * tf.keras.ops.log(1 - z[1] + 1e-4), axis=(1, 2)
        ), name="denoising_loss"
    )([node, p])
    model = tf.keras.Model(inputs=[node, adj], outputs=[loss])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=lambda t, y: y)
    return model


def get_model(base):
    """Ajouter la tête de régression sur les cinq cibles."""
    node = tf.keras.Input(shape = (None, 39), name = "node")
    adj = tf.keras.Input(shape = (None, None, 5), name = "adj")

    x = base([node, adj])
    x = forward(x, 128, rate = 0.4)
    x = L.Dense(5, None)(x)

    model = tf.keras.Model(inputs = [node, adj], outputs = x)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=mcrmse_loss)
    return model

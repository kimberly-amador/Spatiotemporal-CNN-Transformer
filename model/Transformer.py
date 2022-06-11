# Code adapted from: https://github.com/keras-team/keras-io/blob/master/examples/vision/video_transformers.py

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization, Embedding


class PositionalEmbedding(Layer):
    """
    This layer adds positional information to the input tokens.
    """
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs, **kwargs):
        # The inputs are of shape: (batch_size, frames, num_features)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.dense_proj = Sequential([Dense(dense_dim, activation=tf.nn.gelu), Dense(embed_dim), ])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

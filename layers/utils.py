from typing import Optional

import keras


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.embedding_layer = None
        self.dropout_layer = None

    def build(self, input_shape):
        self.embedding_layer = keras.layers.Dense(self.embedding_dim)
        self.dropout_layer = keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        embedding = self.embedding_layer(inputs)
        return self.dropout_layer(embedding, training)


class FeedForwardLayer(keras.layers.Layer):
    def __init__(
        self,
        output_dim: int = 512,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or output_dim * 4
        self.output_dim = output_dim

        self.dense_1 = None
        self.dense_2 = None

    def build(self, input_shape):
        self.dense_1 = keras.layers.Dense(
            self.hidden_dim, activation="relu"
        )
        self.dense_2 = keras.layers.Dense(self.output_dim)

    def call(self, inputs, training):
        dense_outputs = self.dense_1(inputs)
        dense_outputs = self.dense_2(dense_outputs)
        return dense_outputs

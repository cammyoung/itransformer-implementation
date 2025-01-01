import tensorflow as tf
import keras

from layers.encoder import EncoderStack
from layers.utils import EmbeddingLayer, FeedForwardLayer


class EncoderOnly(keras.layers.Layer):
    def __init__(
        self,
        output_len: int,
        output_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
        invert_data: bool = False,
    ):
        super().__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.invert_data = invert_data

        self.invert_layer = None
        self.embedding_layer = None
        self.encoder_stack = None
        self.revert_layer = None

    def build(self, input_shape):
        self.invert_layer = keras.layers.Permute((2, 1))
        self.embedding_layer = EmbeddingLayer(embedding_dim=self.d_model)
        self.encoder_stack = EncoderStack(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout=self.dropout,
        )
        self.projector_layer = keras.layers.Dense(
            self.output_len if self.invert_data else self.output_dim
        )

    def call(self, inputs, training):
        if self.invert_data:
            inputs = self.invert_layer(inputs)

        outputs = self.embedding_layer(inputs, training=training)
        outputs = self.encoder_stack(outputs, training=training)
        outputs = self.projector_layer(outputs)

        if self.invert_data:
            outputs = self.invert_layer(
                outputs
            )[:, :self.output_len, :self.output_dim]

        return outputs


class DualEncoder(keras.layers.Layer):
    def __init__(
        self,
        output_len: int,
        output_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_len = output_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout

        self.embedding_layer = None
        self.encoder = None
        self.invert_layer = None
        self.inverted_embedding_layer = None
        self.inverted_encoder = None
        self.revert_layer = None
        self.dropout_layer = None
        self.feed_forward_layer = None
        self.layer_norm = None
        self.output_layer = None

    def build(self, input_shape):
        self.input_len = input_shape[1]
        self.n_features = input_shape[-1]

        self.embedding_layer = EmbeddingLayer(embedding_dim=self.d_model)
        self.encoder = EncoderStack(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout=self.dropout,
        )

        self.invert_layer = keras.layers.Permute((2, 1))
        self.inverted_embedding_layer = EmbeddingLayer(
            embedding_dim=self.d_model
        )
        self.inverted_encoder = EncoderStack(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout=self.dropout,
        )
        self.inverted_projector_layer = keras.layers.Dense(self.output_len)
        self.revert_layer = keras.layers.Permute((2, 1))

        self.dropout_layer = keras.layers.Dropout(self.dropout)
        self.feed_forward_layer = FeedForwardLayer(
            output_dim=self.n_features, hidden_dim=self.d_model * 4)
        self.layer_norm = keras.layers.LayerNormalization()
        self.output_layer = keras.layers.Dense(self.output_dim)

    def call(self, inputs, training):
        # Encoding all features at each time step of the inputs independently
        embedding = self.embedding_layer(
            inputs, training=training
        )
        outputs = self.encoder(
            [embedding, embedding, embedding],
            training=training
        )  # (B, S, E)

        # Encoding whole sequences of each feature of the inputs independently
        inverted_inputs = self.invert_layer(outputs)  # (B, N, S)
        inverted_embedding = self.inverted_embedding_layer(
            inverted_inputs, training=training
        )  # (B, N, E)
        inverted_outputs = self.inverted_encoder(
            [inverted_embedding, inverted_embedding, inverted_embedding],
            training=training
        )  # (B, N, E)
        inverted_outputs = self.inverted_projector_layer(
            inverted_outputs
        )  # (B, N, S)
        inverted_outputs = self.revert_layer(inverted_outputs)  # (B, S, N)

        # Combine learned representations of time steps and features
        feed_forward_inputs = tf.concat(
            [outputs, inverted_outputs], axis=-1
        )  # (B, S, F + E)
        outputs = self.dropout_layer(feed_forward_inputs, training)
        outputs = self.feed_forward_layer(outputs)  # (B, S, T)

        # Add & norm forecast from inverted encoder
        outputs = self.layer_norm(
            inverted_outputs[:, :, :self.output_dim] + outputs
        )  # (B, S, T)
        outputs = self.output_layer(outputs)  # (B, S, T)

        return outputs

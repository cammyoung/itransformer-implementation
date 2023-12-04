import keras

from .utils import FeedForwardLayer


class EncoderLayer(keras.layers.Layer):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout

        self.attention = None
        self.attention_dropout = None
        self.attention_layer_norm = None
        self.feed_forward = None
        self.feed_forward_dropout = None
        self.layer_norm_2 = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "EncoderLayer expects a tuple of 3 inputs (query, value, key)"
            )

        query_shape, value_shape, key_shape = input_shape

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            dropout=self.dropout,
        )
        # _build_from_signature call required as per
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        self.attention._build_from_signature(
            query_shape, value_shape, key_shape
        )
        self.attention_dropout = keras.layers.Dropout(self.dropout)
        self.layer_norm_1 = keras.layers.LayerNormalization()
        self.feed_forward = FeedForwardLayer(output_dim=self.d_model)
        self.feed_forward_dropout = keras.layers.Dropout(self.dropout)
        self.layer_norm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, training):
        if len(inputs) != 3:
            raise ValueError("EncoderLayer expects a tuple of 3 inputs "
                             "(query, value, key)")

        query, value, key = inputs
        attention_outputs = self.attention(
            query=query,
            value=value,
            key=key,
            attention_mask=None,
            training=training,
        )
        attention_outputs = self.layer_norm_1(
            self.attention_dropout(attention_outputs) + query
        )

        dense_outputs = self.feed_forward(attention_outputs)

        return self.layer_norm_2(
            self.feed_forward_dropout(dense_outputs) + attention_outputs
        )


class EncoderStack(keras.layers.Layer):
    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout

        self.encoder_layers = None

    def build(self, input_shape):
        self.encoder_layers = [
            EncoderLayer(num_heads=self.num_heads,
                         d_model=self.d_model,
                         dropout=self.dropout)
            for _ in range(self.num_layers)
        ]

    def call(self, inputs, training):
        for encoder_layer in self.encoder_layers:
            inputs = encoder_layer(inputs, training=training)
        return inputs

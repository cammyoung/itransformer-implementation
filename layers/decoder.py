import keras

from .utils import FeedForwardLayer


class DecoderLayer(keras.layers.Layer):
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

        self.self_attention = None
        self.self_attention_dropout = None
        self.self_attention_layer_norm = None
        self.cross_attention = None
        self.cross_attention_dropout = None
        self.cross_attention_layer_norm = None
        self.feed_forward
        self.feed_forward_dropout = None
        self.feed_forward_layer_norm = None

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("DecoderLayer expects a tuple of 4 inputs "
                             "(query, value, key, encoder_output)")

        query_shape, value_shape, key_shape, encoder_output_shape = input_shape

        self.self_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            dropout=self.dropout,
        )
        # _build_from_signature call required as per
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        self.self_attention._build_from_signature(
            query_shape, value_shape, key_shape
        )
        self.self_attention_dropout = keras.layers.Dropout(self.dropout)
        self.self_attention_layer_norm = keras.layers.LayerNormalization()

        self.cross_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            dropout=self.dropout,
        )
        # _build_from_signature call required as per
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        self.cross_attention._build_from_signature(
            query_shape, value_shape, encoder_output_shape
        )
        self.cross_attention_dropout = keras.layers.Dropout(self.dropout)
        self.cross_attention_dropout = keras.layers.LayerNormalization()

        self.feed_forward = FeedForwardLayer(output_dim=self.d_model)
        self.feed_forward_dropout = keras.layers.Dropout(self.dropout)
        self.feed_forward_layer_norm = keras.layers.LayerNormalization()

    def call(self, inputs, training):
        if len(inputs) != 4:
            raise ValueError("DecoderLayer expects a tuple of 4 inputs "
                             "(query, value, key, encoder_output)")

        query, value, key, encoder_output = inputs
        self_attention_outputs = self.self_attention(
            query=query,
            value=value,
            key=key,
            attention_mask=None,
            use_causal_mask=True,
            training=training,
        )
        self_attention_outputs = self.self_attention_layer_norm(
            self.self_attention_dropout(self_attention_outputs) + query
        )

        cross_attention_outputs = self.cross_attention(
            query=self_attention_outputs,
            value=encoder_output,
            key=encoder_output,
            attention_mask=None,
            training=training,
        )
        cross_attention_outputs = self.cross_attention_layer_norm(
            self.cross_attention_dropout(cross_attention_outputs)
            + self_attention_outputs
        )

        dense_outputs = self.feed_forward(cross_attention_outputs)

        return self.feed_forward_layer_norm(
            self.feed_forward_dropout(dense_outputs)
            + cross_attention_outputs
        )


class DecoderStack(keras.layers.Layer):
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

        self.decoder_layers = None

    def build(self, input_shape):
        self.decoder_layers = [
            DecoderLayer(num_heads=self.num_heads,
                         d_model=self.d_model,
                         dropout=self.dropout)
            for _ in range(self.num_layers)
        ]

    def call(self, inputs, training):
        for decoder_layer in self.decoder_layers:
            inputs = decoder_layer(inputs, training=training)
        return inputs

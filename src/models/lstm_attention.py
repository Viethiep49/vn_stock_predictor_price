"""
LSTM + Attention Model - Kiến trúc model chính cho dự đoán giá.

Architecture:
- Input(60, 28)
- LSTM(128, return_sequences=True)
- Self-Attention Layer
- LSTM(64, return_sequences=False)
- Dropout(0.2)
- Dense(32, relu)
- Dropout(0.2)
- Dense(1)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class AttentionLayer(layers.Layer):
    """Self-Attention Layer cho time series."""

    def __init__(self, units: int, **kwargs):
        """
        Khởi tạo Attention Layer.

        Args:
            units: Số hidden units
        """
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """Build layer weights."""
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_u",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass.

        Args:
            inputs: Input tensor (batch, time_steps, features)

        Returns:
            Attention-weighted output
        """
        # Score calculation
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), axis=1
        )

        # Weighted sum
        context = inputs * attention_weights
        return context

    def get_config(self):
        """Get layer config for serialization."""
        config = super().get_config()
        config.update({"units": self.units})
        return config


class LSTMAttentionModel:
    """LSTM + Attention model cho stock price prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        num_features: int = 28,
        lstm_units_1: int = 128,
        lstm_units_2: int = 64,
        attention_units: int = 64,
        dropout_rate: float = 0.2,
        dense_units: int = 32,
        learning_rate: float = 0.001,
    ):
        """
        Khởi tạo model.

        Args:
            sequence_length: Số ngày lookback
            num_features: Số features
            lstm_units_1: Units cho LSTM layer 1
            lstm_units_2: Units cho LSTM layer 2
            attention_units: Units cho Attention layer
            dropout_rate: Dropout rate
            dense_units: Units cho Dense layer
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate

        self.model = self._build_model()

    def _build_model(self) -> Model:
        """
        Xây dựng kiến trúc model.

        Returns:
            Compiled Keras Model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))

        # LSTM Layer 1
        x = layers.LSTM(
            units=self.lstm_units_1,
            return_sequences=True,
            name="lstm_1",
        )(inputs)

        # Attention Layer
        x = AttentionLayer(units=self.attention_units, name="attention")(x)

        # LSTM Layer 2
        x = layers.LSTM(
            units=self.lstm_units_2,
            return_sequences=False,
            name="lstm_2",
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Dense layers
        x = layers.Dense(self.dense_units, activation="relu", name="dense_1")(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(1, name="output")(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="lstm_attention")

        # Compile với Huber loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(),
            metrics=["mae", "mse"],
        )

        return model

    def summary(self):
        """In model summary."""
        self.model.summary()

    def get_model(self) -> Model:
        """
        Lấy Keras model.

        Returns:
            Keras Model instance
        """
        return self.model


if __name__ == "__main__":
    # Test model creation
    model = LSTMAttentionModel(
        sequence_length=60,
        num_features=28,
    )
    model.summary()

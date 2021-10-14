import tensorflow as tf
from ..initializers import HeOrthogonal


class Dense(tf.keras.layers.Dense):
    """
    Combines dense layer and scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        use_bias: bool
            True if use bias.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        use_bias=False,
        kernel_initializer=HeOrthogonal(),
        name="dense",
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            name=name,
            **kwargs,
        )

        if isinstance(activation, str):
            activation = activation.lower()
        self._activation = activation
        if self._activation in ["swish", "silu"]:
            self.scale_factor = tf.constant(1 / 0.6, dtype=tf.float32)

    def call(self, x):
        x = super().call(x)
        if self._activation in ["swish", "silu"]:
            x = x * self.scale_factor
        return x


class ResidualLayer(tf.keras.layers.Layer):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
        kernel_initializer: callable
            Initializer of the weight matrices of the dense layers.
    """

    def __init__(
        self,
        units: int,
        nLayers: int = 2,
        activation=None,
        kernel_initializer=HeOrthogonal(),
        name="res",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dense_mlp = tf.keras.models.Sequential(
            [
                Dense(units, 
                    activation=activation, 
                    use_bias=False, 
                    kernel_initializer=kernel_initializer, 
                    name=f"dense_{i}")
                for i in range(nLayers)
            ],
            name="seq",
        )
        self.inv_sqrt_2 = tf.constant(1 / tf.sqrt(2.0), dtype=tf.float32)

    def call(self, inputs):
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x

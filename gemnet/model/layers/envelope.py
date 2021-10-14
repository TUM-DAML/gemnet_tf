import tensorflow as tf
from tensorflow.keras import layers


class Envelope(layers.Layer):
    """
    Envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        p: int
            Exponent of the envelope function.
    """

    def __init__(self, p, name="envelope", **kwargs):
        super().__init__(name=name, **kwargs)
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def call(self, inputs):
        d_scaled = inputs
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return tf.where(d_scaled < 1, env_val, tf.zeros_like(d_scaled))

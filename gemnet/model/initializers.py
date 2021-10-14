import tensorflow as tf


class HeOrthogonal(tf.initializers.Initializer):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """

    def __init__(self, seed: int = None):
        super().__init__()
        self.orth_init = tf.initializers.Orthogonal(seed=seed)

    def __call__(self, shape, dtype=tf.float32):
        W = self.orth_init(shape, dtype)
        fan_in = shape[0]
        if len(shape) == 3:
            fan_in = fan_in * shape[1]

        W = self._standardize(W, shape)
        W *= tf.sqrt(1 / fan_in)  # variance decrease is adressed in the dense layers
        return W

    @staticmethod
    def _standardize(kernel, shape):
        """
        Makes sure that N*Var(W) = 1 and E[W] = 0
        """
        eps = 1e-6
        if len(shape) == 3:
            axis = [0, 1]  # last dimension is output dimension
        else:
            axis = 0
        mean = tf.reduce_mean(kernel, axis=axis, keepdims=True)
        var = tf.math.reduce_variance(kernel, axis=axis, keepdims=True)
        kernel = (kernel - mean) / tf.sqrt(var + eps)
        return kernel

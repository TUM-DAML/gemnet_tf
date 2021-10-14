from ..initializers import HeOrthogonal
import tensorflow as tf


class EfficientInteractionDownProjection(tf.keras.layers.Layer):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size_interm: int,
        kernel_initializer=HeOrthogonal(),
        name="EfficientDownProj",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.emb_size_interm = emb_size_interm
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        num_spherical = input_shape[1][
            2
        ]  # num_spherical is actually num_spherical**2 for quadruplets
        num_radial = input_shape[0][2]
        self.W1 = self.add_weight(
            "kernel",
            shape=[num_spherical, num_radial, self.emb_size_interm],
            trainable=True,
            initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        """
        Returns
        -------
            (rbf_W1, sph): tuple
            - rbf_W1: Tensor, shape=(nEdges, emb_size_interm, num_spherical)
            - sph: Tensor, shape=(nEdges, Kmax, num_spherical)
        """
        tbf = inputs
        rbf_env, sph = tbf  
        # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical) ;  Kmax = maximum number of neighbors of the edges

        # MatMul: mul + sum over num_radial
        rbf_W1 = tf.matmul(rbf_env, self.W1)  # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = tf.transpose(rbf_W1, perm=[1, 2, 0])  # (nEdges, emb_size_interm, num_spherical)
        return rbf_W1, sph


class EfficientInteractionHadamard(tf.keras.layers.Layer):
    """
    Efficient reformulation of the hadamard product and subsequent summation.

    Parameters
    ----------
        emb_size: int
            Embedding size.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        kernel_initializer=HeOrthogonal(),
        name="EfficientHadamard",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        emb_size_interm = input_shape[0][0][1]
        self.W2 = self.add_weight(
            "kernel",
            shape=[self.emb_size, 1, emb_size_interm],
            trainable=True,
            initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        basis, m, id_reduce, Kidx, nEdges = inputs  # quadruplets: m = m_db , triplets: m = m_ba
        # num_spherical is actually num_spherical**2 for quadruplets
        # (nEdges, emb_size_interm, num_spherical) , (nEdges, Kmax, num_spherical)
        rbf_W1, sph = basis

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        # maximum number of neighbors, catch empty id_reduce with maximum
        Kmax = tf.math.maximum(tf.math.reduce_max(Kidx + 1), 0)  
        indices = tf.stack([id_reduce, Kidx], axis=1)
        # (nQuadruplets or nTriplets, emb_size) -> (nEdges, Kmax, emb_size)
        m = tf.scatter_nd(indices, m, shape=(nEdges, Kmax, tf.shape(m)[-1]))

        sum_k = tf.matmul(sph, m, transpose_a=True)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = tf.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # MatMul: mul + sum over emb_size_interm
        m_ca = tf.matmul(self.W2, tf.transpose(rbf_W1_sum_k, perm=[2, 1, 0]))[:, 0]  # (emb_size, nEdges)
        m_ca = tf.transpose(m_ca)  # (nEdges, emb_size)

        return m_ca


class EfficientInteractionBilinear(tf.keras.layers.Layer):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        units_out: int,  # emb_size
        kernel_initializer=HeOrthogonal(),
        name="EfficientBilinear",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.units_out = units_out
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        emb_size_interm = input_shape[0][0][1]
        emb_size = input_shape[1][1]
        self.W2 = self.add_weight(
            "kernel",
            shape=[emb_size, emb_size_interm, self.units_out],
            trainable=True,
            initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        basis, m, id_reduce, Kidx, nEdges = inputs  # quadruplets: m = m_db , triplets: m = m_ba
        # num_spherical is actually num_spherical**2 for quadruplets
        # (nEdges, emb_size_interm, num_spherical) , (nEdges, Kmax, num_spherical)
        rbf_W1, sph = basis

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        # maximum number of neighbors, catch empty id_reduce with maximum
        Kmax = tf.math.maximum(tf.math.reduce_max(Kidx + 1), 0)
        indices = tf.stack([id_reduce, Kidx], axis=1)
        # (nQuadruplets or nTriplets, emb_size) -> (nEdges, Kmax, emb_size)
        m = tf.scatter_nd(indices, m, shape=(nEdges, Kmax, tf.shape(m)[-1]))  

        sum_k = tf.matmul(sph, m, transpose_a=True)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = tf.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # Bilinear: Sum over emb_size_interm and emb_size
        m_ca = tf.matmul(
            tf.transpose(rbf_W1_sum_k, perm=[2, 0, 1]), self.W2
        )  # (emb_size, nEdges, units_out)
        m_ca = tf.math.reduce_sum(m_ca, axis=0)  # (nEdges, units_out)
        return m_ca

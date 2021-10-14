import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .base_layers import Dense
from ..initializers import HeOrthogonal


class AtomEmbedding(layers.Layer):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, name="atom_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        emb_init = tf.initializers.RandomUniform(
            minval=-np.sqrt(3), maxval=np.sqrt(3)
        )
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(93, self.emb_size),
            dtype=self.dtype,
            initializer=emb_init,
            trainable=True,
        )

    def call(self, inputs):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        Z = inputs
        h = tf.gather(self.embeddings, Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(layers.Layer):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(self, emb_size, activation=None, name="edge_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = Dense(
            emb_size, activation=activation, use_bias=False, name="dense"
        )

    def call(self, inputs):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # m_rbf: shape (nEdges, nFeatures)
        # in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_ca
        (
            h,
            m_rbf,
            idnb_a,
            idnb_c,
        ) = inputs  

        h_a = tf.gather(h, idnb_a)  # shape=(nEdges, emb_size)
        h_c = tf.gather(h, idnb_c)  # shape=(nEdges, emb_size)

        m_ca = tf.concat([h_a, h_c, m_rbf], axis=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        return m_ca

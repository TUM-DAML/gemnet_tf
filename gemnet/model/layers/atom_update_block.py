import tensorflow as tf
from .base_layers import ResidualLayer, Dense
from ..initializers import HeOrthogonal
from .scaling import ScalingFactor
from .embedding_block import EdgeEmbedding


class AtomUpdateBlock(tf.keras.layers.Layer):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "atom_update",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dense_rbf = Dense(emb_size_edge, activation=None, use_bias=False, name="MLP_rbf")
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")

        self.layers = self.get_mlp(emb_size_atom, nHidden, activation)

    def get_mlp(self, units, nHidden, activation, name=""):
        dense1 = Dense(units, activation=activation, use_bias=False, name=name + "dense")
        res = [
            ResidualLayer(
                units, nLayers=2, activation=activation, name=name + f"res_{i}"
            )
            for i in range(nHidden)
        ]
        mlp = [dense1] + res
        return mlp

    def call(self, inputs):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        h, m, rbf, id_j = inputs
        nAtoms = tf.shape(h)[0]

        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        x2 = tf.math.unsorted_segment_sum(x, id_j, nAtoms)  # (nAtoms, emb_size_edge)
        x = self.scale_sum(m, x2)

        for i, layer in enumerate(self.layers):
            x = layer(x)  # (nAtoms, emb_size_atom)
        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Activation function to use in the dense layers (except for the final dense layer).
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: str
            Kernel initializer of the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        nHidden: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init="HeOrthogonal",
        scale_file=None,
        name: str = "output",
        **kwargs,
    ):

        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            nHidden=nHidden,
            activation=activation,
            scale_file=scale_file,
            **kwargs,
        )

        if isinstance(output_init, str):
            if output_init.lower() == "heorthogonal":
                output_init = HeOrthogonal()
            elif output_init.lower() == "zeros":
                output_init = tf.zeros_initializer()
            else:
                raise UserWarning(f"Unknown output_init: {output_init}")

        self.direct_forces = direct_forces
        self.dense_rbf = Dense(
            emb_size_edge, activation=None, use_bias=False, name="MLP_rbf"
        )

        self.seq_energy = self.layers  # inherited from parent class
        # do not add bias to final layer to enforce that prediction for an atom 
        # without any edge embeddings is zero
        self.out_energy = Dense(
            num_targets,
            use_bias=False,
            activation=None,
            kernel_initializer=output_init,
            name="dense_energy_final",
        )

        if self.direct_forces:
            self.seq_forces = self.get_mlp(
                emb_size_edge, nHidden, activation, name="F_"
            )
            self.dense_rbf_F = Dense(
                emb_size_edge, activation=None, use_bias=False, name="MLP_rbf_F"
            )
            self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had")
            # no bias in final layer to ensure continuity
            self.out_forces = Dense(
                num_targets,
                use_bias=False, 
                activation=None,
                kernel_initializer=output_init,
                name="dense_forces_final",
            )

    def call(self, inputs):
        """
        Returns
        -------
            (E, F): tuple
            - E: Tensor, shape=(nAtoms, num_targets)
            - F: Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """
        h, m, rbf, id_j = inputs
        nAtoms = tf.shape(h)[0]

        rbf_mlp = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_mlp

        # -------------------------------------- Energy Prediction -------------------------------------- #
        x_E = tf.math.unsorted_segment_sum(x, id_j, nAtoms)  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(m, x_E)

        for i, layer in enumerate(self.seq_energy):
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        x_E = self.out_energy(x_E)  # (nAtoms, num_targets)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:

            x_F = m
            # x_F = self.scale_rbf(m, x)
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            rbf_mlp = self.dense_rbf_F(rbf)  # (nEdges, emb_size_edge)
            x_F = x_F * rbf_mlp
            x_F = self.scale_rbf(m, x_F)

            x_F = self.out_forces(x_F)  # (nEdges, num_targets)
        else:
            x_F = 0
        # ----------------------------------------------------------------------------------------------- #

        return x_E, x_F

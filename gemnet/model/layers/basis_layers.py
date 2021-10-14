import numpy as np
import tensorflow as tf
import sympy as sym

from .envelope import Envelope
from .basis_utils import bessel_basis, real_sph_harm


class BesselBasisLayer(tf.keras.layers.Layer):
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        name="bessel_basis",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_radial = num_radial
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.norm_const = tf.constant(tf.sqrt(2 * self.inv_cutoff), dtype=tf.float32)

        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        def freq_init(shape, dtype):
            return tf.constant(
                np.pi * np.arange(1, shape + 1, dtype=np.float32), dtype=dtype
            )

        self.frequencies = self.add_weight(
            name="frequencies",
            shape=self.num_radial,
            dtype=self.dtype,
            initializer=freq_init,
            trainable=True,
        )

    def call(self, inputs):
        d = inputs  # (nEdges,)
        d = d[:, None]  # (nEdges,1)
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)  
        return env * self.norm_const * tf.sin(self.frequencies * d_scaled) / d


class SphericalBasisLayer(tf.keras.layers.Layer):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient: bool = False,
        name: str = "spherical_basis",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        self.sph_funcs = []  # (num_spherical,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = tf.cast(
            tf.constant(tf.sqrt(self.inv_cutoff ** 3)), dtype=tf.float32
        )

        # convert to tensorflow functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        m = 0  # only single angle
        for l in range(len(Y_lm)):  # num_spherical
            if (
                l == 0
            ):  # Y_00 is only a constant -> function returns value and not tensor
                first_sph = sym.lambdify([theta], Y_lm[l][m], "tensorflow")
                self.sph_funcs.append(
                    lambda theta: tf.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], "tensorflow"))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], "tensorflow")
                )

        self.reshape = tf.keras.layers.Reshape(
            (num_spherical, num_radial)
        )  # needed for efficient reformulation

    def call(self, inputs):
        D_ca, Angle_cab, id3_reduce_ca, Kidx = inputs

        d_scaled = D_ca * self.inv_cutoff  # (nEdges,)
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = tf.stack(rbf, axis=1)  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)

        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = tf.stack(sph, axis=1)  # (nTriplets, num_spherical)

        if not self.efficient:
            rbf_env = tf.gather(
                rbf_env, id3_reduce_ca
            )  # (nTriplets, num_spherical * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # z_ln: l: 0 0  1 1  2 2
            #       n: 0 1  0 1  0 1
            sph = tf.repeat(
                sph, self.num_radial, axis=1
            )  # (nTriplets, num_spherical * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0 0  1 1  2 2
            #       m: 0 0  0 0  0 0
            return rbf_env * sph  # (nTriplets, num_spherical * num_radial)
        else:
            rbf_env = self.reshape(rbf_env)  # (nEdges, num_spherical, num_radial)
            rbf_env = tf.transpose(
                rbf_env, perm=[1, 0, 2]
            )  # (num_spherical, nEdges, num_radial)

            # Zero padded dense matrix
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = tf.math.maximum(tf.math.reduce_max(Kidx + 1), 0)  
            indices = tf.stack([id3_reduce_ca, Kidx], axis=1)
            nEdges = tf.shape(d_scaled)[0]
            sph = tf.scatter_nd(
                indices, sph, shape=(nEdges, Kmax, self.num_spherical)
            )  # (nEdges, Kmax, num_spherical)

            # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical)
            return rbf_env, sph


class TensorBasisLayer(tf.keras.layers.Layer):
    """
    3D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient=False,
        name: str = "tensor_basis",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.efficient = efficient

        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=False
        )
        self.sph_funcs = []  # (num_spherical**2,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = tf.cast(
            tf.constant(tf.sqrt(self.inv_cutoff ** 3)), dtype=tf.float32
        )

        # convert to tensorflow functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        for l in range(len(Y_lm)):  # num_spherical
            for m in range(len(Y_lm[l])):
                if l == 0:
                    # Y_00 is only a constant -> function returns value and not tensor
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], "tensorflow")
                    self.sph_funcs.append(
                        lambda theta, phi: tf.zeros_like(theta) + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], "tensorflow")
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], "tensorflow")
                )

        self.degreeInOrder = (
            tf.range(num_spherical) * 2 + 1
        )  # (num_spherical,) order -l...,0,...l many for each degree l -> 1,3,5,7 etc.
        self.reshape = tf.keras.layers.Reshape((num_spherical, num_radial))
        self.flatten = tf.keras.layers.Reshape(
            (num_spherical ** 2 * num_radial,)
        )  # need to explicitly state shape s.t. tf can infer shape for following layers

    def call(self, inputs):
        D_ca, Alpha_cab, Theta_cabd, id4_reduce_ca, Kidx = inputs

        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)

        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = tf.stack(rbf, axis=1)  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const

        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)
        rbf_env = self.reshape(rbf_env)  # (nEdges, num_spherical, num_radial)
        # atch case where indices are empty
        nEdges = tf.shape(rbf_env)[0]
        rbf_env = tf.cond( nEdges>0,
            lambda: tf.repeat(rbf_env, self.degreeInOrder, axis=1),
            lambda: tf.zeros((0,self.num_spherical**2,self.num_radial)) )  # (nEdges, num_spherical**2, num_radial)
        
        if not self.efficient:
            rbf_env = self.flatten(rbf_env)  # (nEdges, num_spherical**2 * num_radial)
            rbf_env = tf.gather(
                rbf_env, id4_reduce_ca
            )  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # j_ln: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       n: 0  1    0  1  0  1  0  1    0  1  0  1  0  1  0  1  0  1

        sph = [f(Alpha_cab, Theta_cabd) for f in self.sph_funcs]
        sph = tf.stack(sph, axis=1)  # (nQuadruplets, num_spherical**2)

        if not self.efficient:
            sph = tf.repeat(
                sph, self.num_radial, axis=1
            )  # (nQuadruplets, num_spherical**2 * num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0  0    1  1  1  1  1  1    2  2  2  2  2  2  2  2  2  2
            #       m: 0  0   -1 -1  0  0  1  1   -2 -2 -1 -1  0  0  1  1  2  2
            return rbf_env * sph  # (nQuadruplets, num_spherical**2 * num_radial)

        else:
            rbf_env = tf.transpose(
                rbf_env, perm=[1, 0, 2]
            )  # (num_spherical**2, nEdges, num_radial)

            # Zero padded dense matrix
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = tf.math.maximum(tf.math.reduce_max(Kidx + 1), 0)  
            indices = tf.stack([id4_reduce_ca, Kidx], axis=1)
            nEdges = tf.shape(d_scaled)[0]
            sph = tf.scatter_nd(
                indices, sph, shape=(nEdges, Kmax, self.num_spherical ** 2)
            )  # (nEdges, Kmax, num_spherical**2)

            # (num_spherical**2, nEdges, num_radial), (nEdges, Kmax, num_spherical**2)
            return rbf_env, sph

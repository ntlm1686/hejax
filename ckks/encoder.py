# Reference: https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.lax import scan


class Encoder:
    def __init__(self, M: int, scale: float, seed=0):
        """Initialization of the encoder for M a power of 2. 

        xi, which is an M-th root of unity will, be used as a basis for our computations.
        """
        self.xi = jnp.exp(2 * jnp.pi * 1j / M)
        self.M = M
        self.create_sigma_R_basis()
        self.scale = scale
        self.rng_key = jax.random.PRNGKey(seed)

    def vandermonde(self, x: jnp.array, M: int) -> jnp.array:
        N = M // 2
        roots = jnp.power(x, 2 * jnp.arange(N) + 1)
        return jnp.vander(roots, N, increasing=True)

    def sigma_inverse(self, b: jnp.array) -> jnp.array:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        A = self.vandermonde(self.xi, self.M)
        coeffs = jnp.linalg.solve(A, b)
        return coeffs

    def sigma(self, p: jnp.array) -> jnp.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""
        N = self.M // 2
        p = jnp.flip(p)
        roots = jnp.power(self.xi, 2 * jnp.arange(N) + 1)

        _, outputs = scan(
            lambda _, r: (None, jnp.polyval(p, r)), None, roots)
        return outputs

    def pi(self, z: jnp.array) -> jnp.array:
        """Projects a vector of H into C^{N/2}."""
        N = self.M // 4
        return z[:N]

    def pi_inverse(self, z: jnp.array) -> jnp.array:
        """Expands a vector of C^{N/2} by expanding it with its
        complex conjugate."""
        z_conjugate = jnp.flip(jnp.conjugate(z))
        return jnp.concatenate([z, z_conjugate])

    def create_sigma_R_basis(self):
        """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""
        self.sigma_R_basis = self.vandermonde(self.xi, self.M).T

    def compute_basis_coordinates(self, z):
        """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
        R = self.sigma_R_basis
        R_conjugate = jnp.conjugate(R)
        return jnp.real(jnp.matmul(R_conjugate, z.reshape(-1)) / (R_conjugate*R).sum(axis=1))

    def sigma_R_discretization(self, z):
        """Projects a vector on the lattice using coordinate wise random rounding."""
        coordinates = self.compute_basis_coordinates(z)

        rounded_coordinates = coordinate_wise_random_rounding(
            self.rng_key, coordinates)
        y = jnp.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return y

    @partial(jit, static_argnums=(0,))
    def encode(self, z: jnp.array) -> jnp.array:
        """Encodes a vector by expanding it first to H,
        scale it, project it on the lattice of sigma(R), and performs
        sigma inverse.
        """
        pi_z = self.pi_inverse(z)
        scaled_pi_z = self.scale * pi_z
        rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
        p_coef = self.sigma_inverse(rounded_scale_pi_zi)

        return jnp.round(jnp.real(p_coef)).astype(int)

    @partial(jit, static_argnums=(0,))
    def decode(self, p: jnp.array) -> jnp.array:
        """Decodes a polynomial by removing the scale, 
        evaluating on the roots, and project it on C^(N/2)"""
        rescaled_p = p / self.scale
        z = self.sigma(rescaled_p)
        pi_z = self.pi(z)
        return pi_z


def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - jnp.floor(coordinates)
    return coordinates


def coordinate_wise_random_rounding(key, coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = jnp.array([jax.random.choice(key, jnp.array([c, c-1]), p=jnp.array([1-c, c]))
                   for c in r]).reshape(-1)
    rounded_coordinates = coordinates - f
    return rounded_coordinates.astype(int)

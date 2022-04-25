from difflib import context_diff
import jax
import jax.numpy as jnp
from jax import jit

from ckks.utils import get_modulo, shift_mod


@jit
def shift_mod(x, modulo):
    modulo_half = modulo // 2
    return jnp.mod(x + modulo_half, modulo) - modulo_half


def make_polynomial_ring(n: int):
    """Generate a class for ring polynomial in Zq.

    Args:
        n (int): degree of the ring polynomial

    Returns:
        PolynomialRing: a class for ring polynomial in Zq
    """
    modulo = get_modulo(n)

    @jit
    def _mul(x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.polydiv(
            jnp.polymul(x, y),
            modulo
        )[1][-n:].astype(x.dtype),

    def _rmul(x: jnp.array, k: int) -> jnp.array:
        return k * x
    
    _add = jnp.polyadd


    class PolynomialRing(object):
        '''
        Ring-Polynomial: Zq[x] / (x^n + 1)
            range of the reminder is set to (−q/2, q/2]
        '''

        def __init__(self, coeffs: jnp.array, q: int):
            '''
            # Args
                coeffs: coefficients array of a polynomial
                q: modulus
            '''
            self.coeffs = shift_mod(coeffs, q)
            self.q = q

        def __repr__(self):
            template = 'Polynomial ring: {} (mod {}), reminder range: ({}, {}]'
            # TODO numpy
            return template.format(self.coeffs, self.q,
                                   -1-self.q//2, self.q//2)

        def __len__(self):
            # TODO trim leading zeros
            return len(self.coeffs)  # degree of a polynomial

        def __add__(self, other: 'PolynomialRing'):
            coeffs = _add(self.coeffs, other.coeffs)
            return PolynomialRing(coeffs, self.q)

        # def __sub__(self, other: 'PolynomialRing'):
        #     coeffs = _add(self.coeffs, -other.coeffs)
        #     return PolynomialRing(coeffs)

        def __mul__(self, other: 'PolynomialRing'):
            coeffs = _mul(self.coeffs, other.coeffs)
            return PolynomialRing(coeffs, self.q)

        def __rmul__(self, integer: int):
            coeffs = _rmul(self.coeffs, integer)
            return PolynomialRing(coeffs, self.q)

        def __pow__(self, integer):
            if integer == 0:
                return PolynomialRing(jnp.array([1]), self.q)
            ret = self
            for _ in range(integer-1):
                ret *= ret
            return ret

        def round(self):
            return PolynomialRing(jnp.round(self.coeffs.astype(jnp.int64)), self.q)

        @staticmethod
        def sample_gaussian(std: int = 1, mean: int = 0, seed: int = 0):
            jax_key = jax.random.PRNGKey(seed)
            coeffs = jnp.round(std * jax.random.normal(jax_key, (n,)) + mean)
            return PolynomialRing(coeffs)

        @staticmethod
        def sample_uniform(seed: int = 0, minval: int = -q//2, maxval: int = q//2):
            jax_key = jax.random.PRNGKey(seed)
            coeffs = jax.random.randint(
                jax_key, (n,), minval=minval, maxval=maxval)
            return PolynomialRing(coeffs)

    return PolynomialRing

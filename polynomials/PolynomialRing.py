from difflib import context_diff
import jax
import jax.numpy as jnp
from jax import jit

from ckks.utils import get_modulo, shift_mod

def shift_mod(x, modulo):
    modulo_half = modulo // 2
    return jnp.mod(x + modulo_half, modulo) - modulo_half

def make_polynomial_ring(q: int, n: int):
    """Generate a class for ring polynomial in Zq.

    Args:
        q (int): integer modulo
        n (int): degree of the ring polynomial

    Returns:
        PolynomialRing: a class for ring polynomial in Zq
    """
    assert q % 2 == 1, "q must be odd"
    modulo = get_modulo(n)

    @jit
    def _mul(x, y):
        return shift_mod(
            jnp.polydiv(
                jnp.polymul(x, y),
                modulo
            )[1][-n:].astype(x.dtype), 
            q
        )

    @jit
    def _rmul(x, c):
        return shift_mod(
            x * c,
            q
        )

    @jit
    def _add(x, y):
        return shift_mod(
            jnp.polyadd(x, y),
            q
        )

    class PolynomialRing(object):
        '''
        Ring-Polynomial: Zq[x] / (x^n + 1)
            range of the reminder is set to (−q/2, q/2]
        '''

        def __init__(self, coeffs, manual=True):
            '''
            # Args
                coeffs: coefficients array of a polynomial
                q: modulus
            '''
            if manual:
                self.coeffs = shift_mod(coeffs, q)
            else:
                self.coeffs = coeffs

        def __repr__(self):
            template = 'Polynomial ring: {} (mod {}), reminder range: ({}, {}]'
            # TODO numpy
            return template.format(self.coeffs, q,
                                   -1-q//2, q//2)

        def __len__(self):
            # TODO trim leading zeros
            return len(self.coeffs)  # degree of a polynomial

        def __add__(self, other: 'PolynomialRing'):
            coeffs = _add(self.coeffs, other.coeffs)
            return PolynomialRing(coeffs, manual=False)

        def __mul__(self, other: 'PolynomialRing'):
            coeffs = _mul(self.coeffs, other.coeffs)
            return PolynomialRing(coeffs, manual=False)

        def __rmul__(self, integer: int):
            coeffs = _rmul(self.coeffs, integer)
            return PolynomialRing(coeffs, manual=False)

        def __pow__(self, integer):
            if integer == 0:
                return PolynomialRing(jnp.array([1]), manual=False)
            ret = self
            for _ in range(integer-1):
                ret *= ret
            return ret

        @staticmethod
        def sample_gaussian(std: int = 1, mean: int = 0, seed: int = 0):
            jax_key = jax.random.PRNGKey(seed)
            coeffs = jnp.round(std * jax.random.normal(jax_key, (n,)) + mean)
            return PolynomialRing(coeffs)

        @staticmethod
        def sample_uniform(seed: int = 0, minval: int=-q//2, maxval: int=q//2):
            jax_key = jax.random.PRNGKey(seed)
            coeffs = jax.random.randint(jax_key, (n,), minval=minval, maxval=maxval)
            return PolynomialRing(coeffs)

    return PolynomialRing

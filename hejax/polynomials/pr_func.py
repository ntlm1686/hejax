import random
from functools import reduce

import jax
from jax import jit
import jax.numpy as jnp

def get_modulo(N: int) -> jnp.array:
    modulo = jnp.zeros((N+1,)).astype(int)
    modulo = modulo.at[0].set(1)
    modulo = modulo.at[-1].set(1)
    return modulo

def shift_mod(x, modulo):
    modulo_half = modulo // 2
    return jnp.mod(x + modulo_half, modulo) - modulo_half

def make_polynomial_ring_fn(n: int):
    """Generate a class for ring polynomial calculations in Zq[X]/(X^N+1).

    Args:
        n (int): degree of the ring polynomial

    Returns:
        PR_FN: a class consists functions for ring polynomials in Zq[X]/(X^N+1).
    """
    modulo = get_modulo(n)

    @jit
    def _mul(x: jnp.ndarray, y: jnp.ndarray, q: int) -> jnp.ndarray:
        """ Multiply two polynomials in Zq[X]/(X^N+1). """
        return shift_mod(
            jnp.polydiv(
                jnp.polymul(x, y),
                modulo
            )[1][-n:].astype(x.dtype), 
            q
        )

    @jit
    def _add(x: jnp.ndarray, y: jnp.ndarray, q) -> jnp.ndarray:
        """ Add two polynomials in Zq[X]/(X^N+1). """
        return shift_mod(
            jnp.polyadd(x, y),
            q
        )

    class PR_FN:
        @staticmethod
        @jit
        def rmul(q:int, x: jnp.ndarray, k: int) -> jnp.ndarray:
            """ Multiply a polynomial by a scalar. """
            return shift_mod(k * x, q)

        @staticmethod
        @jit
        def mul(q: int, *args) -> jnp.ndarray:
            """ Multiply arbitrary number of polynomials. """
            mul_fn = lambda x, y: _mul(x, y, q)
            return reduce(mul_fn, args)

        @staticmethod
        @jit
        def add(q: int, *args) -> jnp.ndarray:
            """ Add arbitrary number of polynomials. """
            add_fn = lambda x, y: _add(x, y, q)
            return reduce(add_fn, args)

        @staticmethod
        @jit
        def pow(q: int, x: jnp.ndarray, k: int) -> jnp.ndarray:
            """ Power of a polynomial.
            TODO: currently only support k > 1.
            """
            assert k > 0
            ret = x
            for _ in range(k-1):
                ret = _mul(ret, x, q)
            return ret

        @staticmethod
        @jit
        def round(q: int, x: jnp.ndarray) -> jnp.ndarray:
            """ Round the coefficients of a polynomial to the nearest integer. """
            return shift_mod(jnp.round(x), q)

        @staticmethod
        @jit
        def sample_gaussian(q:int, std: int = 1, mean: int = 0, seed: int = 0):
            """ Sample a polynomial from Gaussian distribution. """
            jax_key = jax.random.PRNGKey(seed)
            coeffs = shift_mod(jnp.round(std * jax.random.normal(jax_key, (n,)) + mean), q)
            return coeffs

        @staticmethod
        @jit
        def sample_uniform(q: int, seed: int = 0):
            """ Sample a polynomial from uniform distribution [-q//2, q//2]. """
            jax_key = jax.random.PRNGKey(seed)
            coeffs = jax.random.randint(jax_key, (n,), minval=-q//2, maxval=q//2+1)
            return coeffs

        @staticmethod
        @jit
        def sample_ZO(p: int = 0.5, seed: int = 0):
            """ Sample a polynomial from ZO distribution.
                https://eprint.iacr.org/2016/421.pdf - page 11
            """
            jax_key = jax.random.PRNGKey(seed)
            coeffs = jax.random.choice(jax_key, jnp.array([-1, 0, 1]), (n,), p=jnp.array([p/2, 1-p, p/2]))
            return coeffs

        @staticmethod
        def sample_HWT(h: int, seed: int = 0):
            """ Sample a polynomial from HWT distribution.
                https://eprint.iacr.org/2016/421.pdf - page 11
            """
            random.seed(seed)
            ix = tuple(random.sample(range(n), h))
            coeffs = jnp.zeros(n)
            for i in ix:
                coeffs = coeffs.at[i].set(random.sample([-1,1],1)[0])
            return coeffs

    return PR_FN
from functools import reduce

import jax
import jax.numpy as jnp
from jax import jit

def ring_polymul(poly1, poly2, modulo):
    return jnp.polydiv(jnp.polymul(poly1, poly2), modulo)[1]

def get_modulo(M: int) -> jnp.array:
    modulo = jnp.zeros((M+1,)).astype(int)
    modulo = modulo.at[0].set(1)
    modulo = modulo.at[-1].set(1)
    return modulo

def shift_mod(x, modulo):
    modulo_half = modulo // 2
    return jnp.mod(x + modulo_half, modulo) - modulo_half


# TODO example
_mul = lambda x, y: ring_polymul(x, y, get_modulo(4))[-4:]
def mul(*args):
    return reduce(_mul, args)
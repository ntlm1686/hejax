import jax
import jax.numpy as jnp
from jax import jit

def ring_polymul(x, y, modulo):
    return jnp.polydiv(
        jnp.polymul(x, y),
        modulo
    )

def get_modulo(N: int) -> jnp.array:
    modulo = jnp.zeros((N+1,)).astype(int)
    modulo = modulo.at[0].set(1)
    modulo = modulo.at[-1].set(1)
    return modulo

def shift_mod(x, modulo):
    modulo_half = modulo // 2
    return jnp.mod(x + modulo_half, modulo) - modulo_half
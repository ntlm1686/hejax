import jax
import jax.numpy as jnp

def ring_polymul(poly1, poly2, modulo):
    return jnp.polydiv(jnp.polymul(poly1, poly2), modulo)[1]

def ring_polyadd(poly1, poly2, modulo):
    return jnp.polydiv(jnp.polyadd(poly1, poly2), modulo)[1]

def get_modulo(M: int) -> jnp.array:
    modulo = jnp.zeros((M+1,)).astype(int)
    modulo = modulo.at[0].set(1)
    modulo = modulo.at[-1].set(1)
    return modulo

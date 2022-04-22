import jax
import jax.numpy as jnp
from .utils import (get_modulo, ring_polyadd, ring_polymul)

class Context:
    def __init__(self,
                 M,
                 scale,
                 # ^^^ encoder
                 q, # base modulus
                 p, # base multiplier (prime? TODO)
                 l, # level
                 L, # level limit
                 P, # rescaling parameter
                 # ^^^ encryptor
                 seed=0
                 ):
        jax_key = jax.random.PRNGKey(seed)

        self.M = M
        self.N = M // 2
        self.modulo = get_modulo(self.N)

        self.q = q
        self.Q = q * (p ** L)
        self.p = p

        self.sk = jax.random.randint(
            jax_key, (self.N,), 0, q).astype(int)  # TODO secret key

        self.e = jnp.array([1]) # TODO noise

        a = jax.random.randint(jax_key, (self.N,), 0, self.Q)
        a_s = jnp.mod(ring_polymul(a, self.sk, self.modulo), self.Q)[-self.N:]

        self.pub_key = [
            jnp.mod(ring_polyadd(-a_s, self.e, self.modulo), self.Q),
            a
        ]
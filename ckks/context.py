import jax
import jax.numpy as jnp
from scipy.fftpack import shift
from .utils import (get_modulo, ring_polyadd, ring_polymul, shift_mod)

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
            jax_key, (self.N,), -q//2, q//2).astype(int)  # TODO secret key

        self.sk_square = shift_mod(ring_polymul(self.sk, self.sk, self.modulo), q)

        self.e = jnp.array([0]) # TODO noise

        a = jax.random.randint(jax_key, (self.N,), -self.Q//2, self.Q//2)

        a_s = shift_mod(ring_polymul(a, self.sk, self.modulo), self.Q)[-self.N:]

        self.pub_key = [
            shift_mod(ring_polyadd(-a_s, self.e, self.modulo), self.Q),
            a
        ]
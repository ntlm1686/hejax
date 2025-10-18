import jax
import jax.numpy as jnp
from scipy.fftpack import shift
from .utils import (get_modulo, ring_polymul, shift_mod)
from ..polynomials.PolynomialRing import make_polynomial_ring


class Context:
    def __init__(self,
                 M,
                 scale,
                 # ^^^ encoder
                 q,  # base modulus
                 p,  # base multiplier (prime? TODO)
                 l,  # level
                 L,  # level limit
                 P,  # rescaling parameter
                 # ^^^ encryptor
                 seed=0
                 ):
        self.pr = make_polynomial_ring(q, M//2)
        jax_key = jax.random.PRNGKey(seed)

        self.scale = scale

        self.M = M
        self.N = M // 2
        self.modulo = get_modulo(self.N)

        self.q = q
        self.Q = q * (p ** L)
        self.p = p
        self.P = P

        self.sk = jax.random.randint(
            jax_key, (self.N,), -q//2, q//2).astype(int)  # TODO secret key

        self.sk_square = ring_polymul(
            self.sk, self.sk, self.modulo)[-self.N:]

        self.e = jnp.array([0])  # TODO noise
        self.e0 = jnp.array([0])  # TODO noise

        self.a = jax.random.randint(jax_key, (self.N,), -self.Q//2, self.Q//2)
        a0 = jax.random.randint(jax_key, (self.N,), -self.Q//2*P, self.Q//2*P)

        self.a_s = shift_mod(ring_polymul(
            self.a, self.sk, self.modulo), self.Q)[-self.N:]
        a0_s = shift_mod(ring_polymul(
            a0, self.sk, self.modulo), self.Q*P)[-self.N:]

        self.pub_key = [
            shift_mod(jnp.polyadd(-self.a_s, self.e), self.Q),
            self.a
        ]

        self.evk = [
            shift_mod(
                jnp.polyadd(-a0_s, P * self.sk_square),
                self.Q*P
            ),
            a0
        ]

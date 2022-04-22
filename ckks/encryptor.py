# Reference: Homomorphic Encryption for Arithmetic of Approximate Numbers

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.lax import scan
from typing import List

from .utils import (ring_polyadd, ring_polymul, shift_mod)
from .context import Context


class Encryptor:
    ''' Fully Leveled Homomorphic Encryption '''
    def __init__(self, ctx: Context):
        self.pub_key = ctx.pub_key
        self.modulo = ctx.modulo
        self.N = ctx.N
        self.sk = ctx.sk
        self.Q = ctx.Q
        self.p = ctx.p
        self.q = ctx.q

    @partial(jit, static_argnums=(0,))
    def encrypt(self, message: jnp.array) -> jnp.array:
        ''' TODO (simple case) encrypt a message '''
        return [
            shift_mod(ring_polyadd(self.pub_key[0], message, self.modulo), self.Q),
            self.pub_key[1]
        ]

    @partial(jit, static_argnums=(0,))
    def decrypt(self,
                ciphertext: List[jnp.array],
                l: int # level of ciphertext
                ) -> jnp.array:
        ''' TODO decrypt a message '''
        return shift_mod(
            jnp.polyadd(
                ciphertext[0],
                ring_polymul(ciphertext[1], self.sk, self.modulo)
            )[-self.N:], # trim zeros (jax issue)
            (self.p**l) * self.q
            )
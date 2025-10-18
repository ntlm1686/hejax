import jax
import jax.numpy as jnp
from typing import List

from .utils import get_modulo


def cipheradd(cipher1, cipher2, modulo):
    return [
        jnp.polydiv(jnp.polyadd(cipher1[0],cipher2[0]), modulo)[1],
        jnp.polydiv(jnp.polyadd(cipher1[0],cipher2[0]), modulo)[1],
    ]

def ciphersub(cipher1, cipher2, modulo):
    return [
        jnp.polydiv(jnp.polysub(cipher1[0],cipher2[0]), modulo)[1],
        jnp.polydiv(jnp.polysub(cipher1[0],cipher2[0]), modulo)[1],
    ]

def ciphermul(cipher1, cipher2, modulo):
    return [
        jnp.polydiv(jnp.polymul(cipher1[0],cipher2[0]), modulo)[1],
        jnp.polydiv(jnp.polymul(cipher1[0],cipher2[1]), modulo)[1] + \
        jnp.polydiv(jnp.polymul(cipher1[1],cipher2[0]), modulo)[1],
        jnp.polydiv(jnp.polymul(cipher1[1],cipher2[1]), modulo)[1],
    ]

def ciphermul_constant(cipher, constant, modulo):
    return [
        jnp.polydiv(jnp.polymul(cipher[0], constant), modulo)[1],
        jnp.polydiv(jnp.polymul(cipher[1], constant), modulo)[1],
    ]

def relinearize(c_mult, evk, p, modulo):
    P = [
        1/p * jnp.polydiv(jnp.polymul(c_mult[2], evk[0]), modulo)[1],
        1/p * jnp.polydiv(jnp.polymul(c_mult[2], evk[1]), modulo)[1]
    ]
    return [
        jnp.polyadd(jnp.polydiv(jnp.polymul(c_mult[0],P), modulo)[1], P[0]),
        jnp.polyadd(jnp.polydiv(jnp.polymul(c_mult[1],P), modulo)[1], P[1]),
    ]

class Cipher:
    _add = staticmethod(cipheradd)
    _sub = staticmethod(ciphersub)
    _mul = staticmethod(ciphermul)
    _mul_constant = staticmethod(ciphermul_constant)
    _relin = staticmethod(relinearize)

    def __init__(self, content: List[jnp.array], modulo) -> None:
        self.content = content
        if isinstance(modulo, jnp.ndarray):
            self.modulo = modulo
        elif isinstance(modulo, int):
            self.modulo = get_modulo(modulo)
        self.depth = 1


    def __mul__(self, other):
        if isinstance(other, Cipher):
            c_mult = self._mul(self, self.content, other.content, self.modulo)
            content = self._relin(c_mult, self.evk)
        elif isinstance(other, jnp.array):
            content = self._mul_constant(self.content, other, self.modulo)
        return self.__class__(content, self.modulo)
    
    def __add__(self, other):
        if isinstance(other, Cipher):
            content = self._add(self.content, other.content, self.modulo)
        else:
            raise NotImplementedError
        return self.__class__(content, self.modulo)
    
    def __sub__(self, other):
        if isinstance(other, Cipher):
            content = self._sub(self.content, other.content, self.modulo)
        else:
            raise NotImplementedError
        return self.__class__(content, self.modulo)
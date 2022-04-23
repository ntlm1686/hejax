import numpy as np
import jax.numpy as jnp
from .utils import crange


class RingPolynomial(object):
    '''
    Ring-Polynomial: Zq[x] / (x^n + 1)
        range of the reminder is set to (−q/2, q/2]
    '''
    def __init__(self, coeffs, q):
        '''
        # Args
            coeffs: coefficients array of a polynomial
            q: modulus
        '''
        n = len(coeffs)  # degree of a polynomial

        f = np.zeros((n+1), dtype=np.int64)  # x^n + 1
        f[0] = f[-1] = 1
        f = np.poly1d(f)
        self.f = f

        self.q = q
        coeffs = np.array(coeffs, dtype=np.int64) % q
        coeffs = crange(coeffs, q)
        self.poly = np.poly1d(np.array(coeffs, dtype=np.int64))

    def __repr__(self):
        template = 'Rq: {} (mod {}), reminder range: ({}, {}]'
        return template.format(self.poly.__repr__(), self.q,
                               -self.q//2, self.q//2)

    def __len__(self):
        return len(self.poly)  # degree of a polynomial

    def __add__(self, other):
        coeffs = np.polyadd(self.poly, other.poly).coeffs
        return RingPolynomial(coeffs, self.q)

    def __mul__(self, other):
        q, r = np.polydiv(np.polymul(self.poly, other.poly), self.f)
        coeffs = r.coeffs
        return RingPolynomial(coeffs, self.q)

    def __rmul__(self, integer):
        coeffs = (self.poly.coeffs * integer)
        return RingPolynomial(coeffs, self.q)

    def __pow__(self, integer):
        if integer == 0:
            return RingPolynomial([1], self.q)
        ret = self
        for i in range(integer-1):
            ret *= ret
        return ret

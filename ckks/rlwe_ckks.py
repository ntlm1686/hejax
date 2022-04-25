import jax.numpy as jnp

from typing import List

from polynomials.PolynomialRing import make_polynomial_ring



class RLWECKKS:
    def __init__(self, n, q, P, std, seed=0):
        self.n = n
        self.P = P
        self.std = std
        self.PR = make_polynomial_ring(q, n) # Z_q[X]
        self.PR_evk = make_polynomial_ring(q*P, n) # Z_Pq[X]
        self.seed = seed

    def generate_keys(self):
        s = self.PR.sample_gaussian(std=self.std, seed=self.seed)
        e = self.PR.sample_gaussian(std=self.std, seed=self.seed)

        a = self.PR.sample_uniform(seed=self.seed)
        b = -1 * a * s + e

        s_ = self.PR_evk(s.coeffs) # cast to Z_Pq[X]  
        s_square = self.PR_evk((s**2).coeffs) # cast to Z_Pq[X]
        a_ = self.PR_evk.sample_uniform(seed=self.seed)
        e_ = self.PR_evk.sample_gaussian(std=self.std, seed=self.seed)

        self.evk = (
            -1 * a_ * s_ + self.P * s_square + e_,
            a_
        )

        self.s = s
        self.pk =(b, a)

        return (s, (b, a))  # (secret, public)

    def encrypt(self, m: jnp.array, pk):
        '''
        # Args:
            m: plaintext (mod t)
            a: public key (a0, a1)
        '''
        e0 = self.PR.sample_gaussian(std=self.std, seed=self.seed)
        e1 = self.PR.sample_gaussian(std=self.std, seed=self.seed)

        r = self.PR(jnp.array([1]))
        m = self.PR(m)

        return (
            r * pk[0] + e0 + m,
            r * pk[1] + e1
        )

    def decrypt(self, c, sk):
        '''
        # Args:
            c: ciphertext (c0, c1, ..., ck)
            s: secret key
        '''
        c = [self.PR(c_i.coeffs) for c_i in c]

        c = [ci * sk**i for i, ci in enumerate(c)]

        m = c[0]
        for i in range(1, len(c)):
            m += c[i]

        return m

    def add(self, c0, c1):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        return (
            c0[0] + c1[0],
            c0[1] + c1[1]
        )

    def mul(self, c0, c1):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        c = ()

        k0 = len(c0) - 1
        k1 = len(c1) - 1

        for _ in range(k1):
            c0 += (self.PR(jnp.array([0])),)

        for _ in range(k0):
            c1 += (self.PR(jnp.array([0])),)

        for i in range(k0 + k1 + 1):
            _c = self.PR(jnp.array([0]))
            for j in range(i+1):
                _c += c0[j] * c1[i-j]
            c += (_c,)

        return c

    def relinear(self, d):
        c0 = d[0] + (1/self.P * d[2] * self.evk[0]).round() # Z_q[x] * Z_Pq[x] -> Z_q[X]
        c1 = d[1] + (1/self.P * d[2] * self.evk[1]).round()
        return c0, c1
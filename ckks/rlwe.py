import jax.numpy as jnp
from polynomials.PolynomialRing import make_polynomial_ring


class RLWE:
    def __init__(self, n, p, t, std, seed=0):
        self.n = n
        self.p = p
        self.t = t
        self.std = std
        self.PR_c = make_polynomial_ring(p, n)
        self.PR_p = make_polynomial_ring(t, n)
        self.seed = seed

    def generate_keys(self):
        s = self.PR_c.sample_gaussian(std=self.std, seed=self.seed)
        # e = self.PR_c.sample_gaussian(std=self.std, seed=self.seed)
        e = self.PR_c(jnp.array([1]))

        a = self.PR_c.sample_uniform(seed=self.seed)
        b = -1 * (a * s + self.t * e)

        return (s, (b, a))  # (secret, public)

    def encrypt(self, m, pk):
        '''
        # Args:
            m: plaintext (mod t)
            a: public key (a0, a1)
        '''
        b, a = pk
        # e = [self.PR_c.sample_gaussian(std=self.std, seed=self.seed)
        #      for _ in range(3)]
        e = [self.PR_c(jnp.array([1]))
             for _ in range(3)]

        m = self.PR_c(m)

        return (
            m + b * e[0] + self.t * e[2],
                a * e[0] + self.t * e[1]
        )

    def decrypt(self, c, s):
        '''
        # Args:
            c: ciphertext (c0, c1, ..., ck)
            s: secret key
        '''
        c = [ci * s**i for i, ci in enumerate(c)]

        m = c[0]
        for i in range(1, len(c)):
            m += c[i]

        m = self.PR_p(m.coeffs)

        return m

    def add(self, c0, c1):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        c = ()

        k0 = len(c0)  # not necessary to compute (len - 1)
        k1 = len(c1)

        if k0 > k1:
            (c0, c1) = (c1, c0)  # c0 is always shorter

        for _ in range(abs(k0 - k1)):
            c0 += (self.PR_c(jnp.array([0])),)  # add 0 to shorter ciphertext

        for i in range(len(c0)):
            c += (c0[i] + c1[i],)

        return c

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
            c0 += (self.PR_c(jnp.array([0])),)

        for _ in range(k0):
            c1 += (self.PR_c(jnp.array([0])),)

        for i in range(k0 + k1 + 1):
            _c = self.PR_c(jnp.array([0]))
            for j in range(i+1):
                _c += c0[j] * c1[i-j]
            c += (_c,)

        return c

from typing import List

import jax.numpy as jnp

from polynomials.pr_func import make_polynomial_ring_fn


class Encryptor:
    def __init__(self,
                 n,      # number of coefficients
                 q,      # base modulus
                 p,      # modulus multiplier
                 L,      # multiplicative depth
                 P,      # relinearization scale
                 std,    # noise standard deviation
                 h,      # secret key hamming weight
                 seed=0):
        self.n = n
        self.P = P
        self.Q = q * p**L
        self.h = h
        self.q = q
        self.p = p

        self.std = std
        self.pr_fn = make_polynomial_ring_fn(n)
        self.seed = seed

    def generate_keys(self):
        s = self.pr_fn.sample_HWT(h=self.h, seed=self.seed)  # secret key
        e = self.pr_fn.sample_gaussian(
            self.Q, std=self.std, seed=self.seed)  # noise
        # sample from Z_Q[X]/[X^N+1]
        a = self.pr_fn.sample_uniform(self.Q, seed=self.seed)

        b = self.pr_fn.add(self.Q,
                           self.pr_fn.rmul(
                               self.Q, self.pr_fn.mul(self.Q, a, s), -1),
                           e
                           )

        # s_ = self.PR_evk(s.coeffs)  # cast to Z_Pq[X]
        # s_square = self.PR_evk((s**2).coeffs)  # cast to Z_Pq[X]
        # a_ = self.PR_evk.sample_uniform(seed=self.seed)
        # e_ = self.PR_evk.sample_gaussian(std=self.std, seed=self.seed)

        # self.evk = (
        #     -1 * a_ * s_ + self.P * s_square + e_,
        #     a_
        # )
        return (s, (b, a))  # (secret, public)

    def encrypt(self, m: jnp.ndarray, pk: List[jnp.ndarray]):
        '''
        # Args:
            m: plaintext (mod t)
            a: public key (a0, a1)
        '''
        e0 = self.pr_fn.sample_gaussian(self.Q, std=self.std, seed=self.seed)
        e1 = self.pr_fn.sample_gaussian(self.Q, std=self.std, seed=self.seed)
        v = self.pr_fn.sample_ZO(self.Q, seed=self.seed)

        return (
            self.pr_fn.add(self.Q,
                           self.pr_fn.mul(self.Q, pk[0], v), e0, m
                           ),
            self.pr_fn.add(self.Q,
                           self.pr_fn.mul(self.Q, pk[1], v), e1
                           )
        )

    def decrypt(self, c, l, sk):
        '''
        # Args:
            c: ciphertext (c0, c1, ..., ck)
            s: secret key
        '''
        # c = [ci * sk**i for i, ci in enumerate(c)]
        c = [
            self.pr_fn.mul(self.q * self.p**l,
                ci,
                self.pr_fn.pow(self.q * self.p**l,
                    sk, i
                )
            ) for i, ci in enumerate(c)
        ]

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
        # Z_q[x] * Z_Pq[x] -> Z_q[X]
        c0 = d[0] + (1/self.P * d[2] * self.evk[0]).round()
        c1 = d[1] + (1/self.P * d[2] * self.evk[1]).round()
        return c0, c1

    def rescale(self, d):
        pass

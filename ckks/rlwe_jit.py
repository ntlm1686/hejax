from typing import List
from functools import partial

import jax.numpy as jnp
from jax import jit

from polynomials.pr_func import make_polynomial_ring_fn


class Encryptor:
    def __init__(self,
                 n,      # number of coefficients
                 q,      # base modulus
                 p,      # modulus multiplier
                 L,      # max multiplicative depth
                 P,      # relinearization scale
                 std,    # noise standard deviation
                 h,      # secret key hamming weight
                 seed=0):
        self.n = n
        self.P = P
        self.h = h
        self.q = q
        self.p = p
        self.std = std
        self.seed = seed
        self.pr_fn = make_polynomial_ring_fn(n)

        self.Q = q * p**L  # q_L
        self.q_ = [q * p**l for l in range(L+1)]

    def generate_keys(self):
        s = self.pr_fn.sample_HWT(h=self.h, seed=self.seed)  # secret key
        e = self.pr_fn.sample_gaussian(
            self.Q, std=self.std, seed=self.seed)  # noise
        # sample from Z_Q[X]/[X^N+1]
        a = self.pr_fn.sample_uniform(self.Q, seed=self.seed)

        b = self.pr_fn.add(self.Q,
                           -1 * self.pr_fn.mul(self.Q, a, s), e)

        ss = self.pr_fn.mul(self.Q, s, s)
        a_ = self.pr_fn.sample_uniform(self.P*self.Q, seed=self.seed)
        e_ = self.pr_fn.sample_gaussian(self.Q, std=self.std, seed=self.seed)

        self.evk = (
            self.pr_fn.add(self.P*self.Q,
                           -1 * self.pr_fn.mul(self.P*self.Q, a_, s),
                           self.P * ss,
                           e_
                           ),
            a_
        )
        return (s, (b, a))  # (secret key, public key)

    @partial(jit, static_argnums=(0,))
    def encrypt(self, m: jnp.ndarray, pk: List[jnp.ndarray]):
        '''
        # Args:
            m: plaintext (mod t)
            a: public key (a0, a1)
        '''
        e0 = self.pr_fn.sample_gaussian(self.Q, std=self.std, seed=self.seed)
        e1 = self.pr_fn.sample_gaussian(self.Q, std=self.std, seed=self.seed)
        v = self.pr_fn.sample_ZO(seed=self.seed)

        return (
            self.pr_fn.add(self.Q,
                           self.pr_fn.mul(self.Q, pk[0], v), e0, m),
            self.pr_fn.add(self.Q,
                           self.pr_fn.mul(self.Q, pk[1], v), e1)
        )

    @partial(jit, static_argnums=(0, 2))
    def decrypt(self, c, l, sk):
        '''
        # Args:
            c: ciphertext (c0, c1, ..., ck)
            s: secret key
        '''
        q = self.q_[l]
        return self.pr_fn.add(q,
                              c[0],
                              self.pr_fn.mul(q, c[1], sk))

    @partial(jit, static_argnums=(0, 3))
    def add(self, c0, c1, l):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        return (
            self.pr_fn.add(self.q_[l], c0[0], c1[0]),
            self.pr_fn.add(self.q_[l], c0[1], c1[1]),
        )

    def _mul(self, c0, c1, l):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        q = self.q_[l]
        return (
            self.pr_fn.mul(q, c0[0], c1[0]),
            self.pr_fn.add(q,
                           self.pr_fn.mul(q, c0[0], c1[1]),
                           self.pr_fn.mul(q, c0[1], c1[0]),
                           ),
            self.pr_fn.mul(q, c0[1], c1[1]),
        )

    def _relinear(self, d, l):
        x = d[2]/self.P
        q = self.q_[l]
        cc0 = self.pr_fn.add(q,
                             d[0],
                             jnp.round(self.pr_fn.mul(
                                 q, x, self.evk[0])),  # TODO mod?
                             )
        cc1 = self.pr_fn.add(q,
                             d[1],
                             jnp.round(self.pr_fn.mul(
                                 q, x, self.evk[1])),
                             )
        return (cc0, cc1)

    @partial(jit, static_argnums=(0, 3))
    def mul(self, c0, c1, l):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        d = self._mul(c0, c1, l)
        d = self._relinear(d, l)
        return d

    def rescale(self, d, lo, ln):
        q = self.q_[ln]
        scale = self.p**(ln - lo)
        return (
            self.pr_fn.round(q, scale * d[0]),
            self.pr_fn.round(q, scale * d[1]),
        )
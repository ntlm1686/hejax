import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

class CKKSEncoder:
    
    def __init__(self, M: int):
        """Initialization of the encoder for M a power of 2. 
        
        xi, which is an M-th root of unity will, be used as a basis for our computations.
        """
        self.xi = jnp.exp(2 * jnp.pi * 1j / M)
        self.M = M
        
    @partial(jit, static_argnums=(0,))
    def sigma_inverse(self, b: jnp.array) -> jnp.array:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        N = self.M // 2
        root = self.xi
        roots = jnp.power(root, 2 * jnp.arange(N) + 1)
        A = jnp.vander(roots, N)

        # Then we solve the system
        coeffs = jnp.linalg.solve(A, b)

        # Finally we output the polynomial
        return coeffs

    @partial(jit, static_argnums=(0,))
    def sigma(self, p: jnp.array) -> jnp.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""

        outputs = []
        N = self.M //2

        # We simply apply the polynomial on the roots
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = jnp.polyval(p, root)
            outputs.append(output)
        return jnp.array(outputs)
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

def compare(f_jax, f_numpy, N=4096, M=100, args: list = []):
    data_gpu = [jax.random.uniform(jax.random.PRNGKey(1), (N,)).astype(jnp.complex128) for _ in range(M)]
    data_cpu = [np.random.rand(N).astype(np.complex128) for _ in range(M)]

    @jit
    def fn_gpu():
        for arg in data_gpu:
            f_jax(arg, *args)

    def fn_cpu():
        for arg in data_cpu:
            f_numpy(arg, *args)
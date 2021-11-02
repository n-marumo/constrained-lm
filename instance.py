import jax
import jax.numpy as jnp

import mymodule


class QuadraticL1ball3:
    def __init__(self, d, r, n, nnz, xmax, seed):
        self.n = n
        self.d = d
        self.r = r
        self.seed = seed
        key = jax.random.PRNGKey(seed)

        self.a = jax.random.normal(key, (n, r, d)) / jnp.sqrt(r)
        self.b = jax.random.normal(key, (n, d))

        self.x_true = jax.ops.index_update(
            jnp.zeros(d),
            jax.random.choice(key, d, (nnz,)),
            (jax.random.uniform(key, (nnz,)) * 2 - 1) * xmax,
        )
        self.y = self.f(self.x_true)
        self.r_ball = jnp.linalg.norm(self.x_true, ord=1)

        self.dim_domain = d
        self.dim_range = n
        self.x0 = jnp.zeros(d)

        # print(self.x_true)

    def f(self, x):
        ax = self.a @ x
        return jnp.einsum("ij, ij -> i", ax, ax) / 2 + self.b @ x

    def inner_func(self, x):
        return self.y - self.f(x)

    def outer_func(self, r):
        return mymodule.sqnorm(r) / 2

    def proj(self, x):
        return mymodule.projection_l1_ball_jax(x, self.r_ball)


class NMF:
    def __init__(self, m, n, r, p, cond_param, noise_init, seed):
        self.m = m
        self.n = n
        self.r = r
        self.p = p
        self.seed = seed
        key = jax.random.PRNGKey(seed)

        mn = min(m, n)
        y = (
            jax.random.uniform(key, (m, mn))
            @ (jnp.geomspace(1, 1 / cond_param, mn).reshape(-1, 1) * jax.random.uniform(key, (n, mn)).T)
        ).reshape(-1)
        y /= jnp.amax(y)
        self.c = jax.random.bernoulli(key, p, (m * n,))
        self.y = y[self.c]

        self.dim_domain = (m + n) * r
        self.dim_range = jnp.sum(self.c)
        self.x0 = jax.random.uniform(key, (self.dim_domain,)) * noise_init

    def inner_func(self, x):
        u, v = jnp.split(x.reshape(-1, self.r), [self.m])
        return (u @ v.T).reshape(-1)[self.c] - self.y

    def outer_func(self, r):
        return mymodule.sqnorm(r) / 2

    def proj(self, x):
        return jnp.maximum(x, 0)

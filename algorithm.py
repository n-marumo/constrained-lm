import functools
import time

import jax
import jax.numpy as jnp
import scipy.linalg
import scipy.optimize as spopt

import mymodule

TIMEOUT = 20
ETA_GRADMAP = 1e8


def norm_gradmap_fixedeta(x, func, proj):
    grad = jax.grad(func)
    return jnp.linalg.norm(ETA_GRADMAP * (x - proj(x - grad(x) / ETA_GRADMAP)))


def norm_gradmap_fixedeta_with_grad(x, grad, proj):
    return jnp.linalg.norm(ETA_GRADMAP * (x - proj(x - grad / ETA_GRADMAP)))


@mymodule.func_name
def scipy_optimize_NMF_2(instance, method="trf"):
    def func(x):
        return mymodule.sqnorm(instance.inner_func(x)) / 2

    start_time = time.time()
    res = spopt.least_squares(
        fun=instance.inner_func,
        x0=instance.x0.reshape(-1),
        bounds=(0, jnp.inf),
        method=method,
        # verbose=2,
    )
    # print(res.status)
    elapsed = time.time() - start_time
    x = res.x
    f = func(x).item()
    norm_gm = norm_gradmap_fixedeta(x, func, instance.proj).item()
    result = [
        elapsed,
        f,
        norm_gm,
    ]
    print(result)
    return jnp.array(result).reshape(1, -1)


def apg_restart(
    func,
    func_grad,
    proj,
    x0,
    mu,
    eta0,
    maxiter=100000,
    tol_gradmap=0,
    tol_f=0,
    eta_inc=2,
    eta_dec=0.9,
):
    # initialization
    x_pre = x = x0
    alpha_pre = 1
    eta = jnp.maximum(eta0, mu)
    k = 0
    func_x = func(x)
    restarted = False

    while k < maxiter:
        alpha = jnp.sqrt(mu / eta)
        y = x + (x - x_pre) * alpha / alpha_pre * (1 - alpha_pre) / (1 + alpha)
        func_y, grad_y = func_grad(y)
        z = proj(y - grad_y / eta)
        func_z = func(z)
        sqnorm_zy = mymodule.sqnorm(z - y)

        # backtracking for eta
        if func_z > func_y + jnp.vdot(grad_y, z - y) + eta / 2 * sqnorm_zy:
            eta *= eta_inc
            continue

        # adaptive restart
        # see Section 3.2 in [B. O'Donoghue and E. Candes (2015)]
        if func_z > func_x:
            if restarted:
                break
            # print(f"Restart at k = {k}")
            x_pre = x
            alpha_pre = 1
            restarted = True
            continue
        else:
            restarted = False

        x_pre = x
        x = z
        func_x = func_z
        alpha_pre = alpha
        k += 1

        if sqnorm_zy <= (tol_gradmap / eta) ** 2 or func_x <= tol_f:
            break

        eta = jnp.maximum(eta * eta_dec, mu)

    # print(f"APG stopped in {k} iterations")
    return x, func_x, eta


@mymodule.func_name
def projected_gradient(
    instance,
    eta0=1e-0,
    eta_inc=2,
    eta_dec=0.9,
    tol_gradmap=0,
    maxiter=100000,
    timeout=100000,
    print_interval=10,
    return_sol=False,
):
    def func(x):
        return instance.outer_func(instance.inner_func(x))

    grad = jax.grad(func)

    def record_print_results():
        nonlocal offset_time
        record_start = time.time()
        results.append(
            [
                time.time() - start_time - offset_time,
                func_x,
                norm_gradmap_fixedeta(x, func, instance.proj),
                eta,
            ]
        )
        if return_sol:
            sols.append(x)
        if print_interval is not None and k % print_interval == 0:
            print(k, results[-1][:])
            # print(x)
        offset_time += time.time() - record_start

    # initialization
    results = []
    sols = []
    x = instance.x0
    eta = eta0  # inverse step-size
    start_time = time.time()
    offset_time = 0
    func_x = func(x)
    k = 0
    record_print_results()
    grad_x = grad(x)

    # main loop
    while k < maxiter:
        y = instance.proj(x - grad_x / eta)
        func_y = func(y)

        # backtracking for eta
        if func_y > func_x + jnp.vdot(grad_x, y - x) + eta / 2 * mymodule.sqnorm(y - x):
            eta *= eta_inc
            continue

        x = y
        func_x = func_y
        eta *= eta_dec
        k += 1

        record_print_results()
        if results[-1][0] > timeout or results[-1][2] <= tol_gradmap:
            break

        grad_x = grad(x)

    print(k, results[-1])
    if return_sol:
        return results, sols
    else:
        return results


@mymodule.func_name
def proposed_lm_jvp(
    instance,
    M0=1e-0,
    M_inc=2,
    M_dec=0.9,
    M_min=1e-15,
    eta0=1e-0,
    eta_inc=2,
    eta_dec=0.9,
    tol_gradmap=0,
    maxiter=100000,
    maxiter_inner=100000,
    tol_coef_inner=0,
    timeout=100000,
    print_interval=10,
    return_sol=False,
):
    def func(x):
        return mymodule.sqnorm(instance.inner_func(x)) / 2

    def model(xk, lam):
        F, F_vjp = jax.vjp(instance.inner_func, xk)
        F_jvp = functools.partial(jax.jvp, instance.inner_func, (xk,))

        def func(x):
            r = F + F_jvp((x - xk,))[1]
            return (mymodule.sqnorm(r) + lam * mymodule.sqnorm(x - xk)) / 2

        def func_grad(x):
            r = F + F_jvp((x - xk,))[1]
            val = (mymodule.sqnorm(r) + lam * mymodule.sqnorm(x - xk)) / 2
            grad = F_vjp(r)[0] + lam * (x - xk)
            return val, grad

        return func, func_grad

    def record_print_results():
        nonlocal offset_time
        record_start = time.time()
        results.append(
            [
                time.time() - start_time - offset_time,
                func_x,
                norm_gradmap_fixedeta(x, func, instance.proj),
                eta,
                M,
            ]
        )
        if return_sol:
            sols.append(x)
        if print_interval is not None and k % print_interval == 0:
            print(k, results[-1])
            # print(x)
        offset_time += time.time() - record_start

    # initialization
    results = []
    sols = []
    x = instance.x0
    M = M0
    eta = eta0
    start_time = time.time()
    offset_time = 0
    func_x = func(x)
    k = 0
    record_print_results()

    while k < maxiter:
        norm_F = jnp.sqrt(2 * func_x)
        lam = M * norm_F
        eta = jnp.maximum(eta, lam)
        model_func, model_func_grad = model(x, lam)
        x_new, model_x_new, eta = apg_restart(
            model_func,
            model_func_grad,
            instance.proj,
            x,
            lam,
            eta,
            maxiter=maxiter_inner,
            tol_gradmap=tol_coef_inner * lam * norm_F,
            eta_inc=eta_inc,
            eta_dec=eta_dec,
        )
        func_new = func(x_new)

        # backtracking for M
        if func_new > model_x_new:
            M *= M_inc
            continue

        x = x_new
        func_x = func_new
        M = max(M_dec * M, M_min)
        k += 1

        record_print_results()
        if results[-1][0] > timeout or results[-1][2] <= tol_gradmap:
            break

    print(k, results[k])
    if return_sol:
        return results, sols
    else:
        return results


@mymodule.func_name
def fan2013_jvp(
    instance,
    mu=1e-4,
    beta=0.9,
    sigma=1e-4,
    gamma=0.99995,
    delta=1,
    eta0=1e-0,
    eta_inc=2,
    eta_dec=0.9,
    tol_gradmap=0,
    tol_gradmap_inner=0,
    maxiter=100000,
    maxiter_inner=100000,
    timeout=100000,
    print_interval=10,
    return_sol=False,
):
    def func(x):
        return mymodule.sqnorm(instance.inner_func(x)) / 2

    grad = jax.grad(func)

    def model(xk, lam):
        F, F_vjp = jax.vjp(instance.inner_func, xk)
        F_jvp = functools.partial(jax.jvp, instance.inner_func, (xk,))

        def func(x):
            r = F + F_jvp((x - xk,))[1]
            return (mymodule.sqnorm(r) + lam * mymodule.sqnorm(x - xk)) / 2

        def func_grad(x):
            r = F + F_jvp((x - xk,))[1]
            val = (mymodule.sqnorm(r) + lam * mymodule.sqnorm(x - xk)) / 2
            grad = F_vjp(r)[0] + lam * (x - xk)
            return val, grad

        return func, func_grad

    def record_print_results():
        nonlocal offset_time
        record_start = time.time()
        results.append(
            [
                time.time() - start_time - offset_time,
                func_x,
                norm_gradmap_fixedeta(x, func, instance.proj),
                eta,
            ]
        )
        if return_sol:
            sols.append(x)
        if print_interval is not None and k % print_interval == 0:
            print(k, results[-1][:])
            # print(x)
        offset_time += time.time() - record_start

    # initialization
    results = []
    sols = []
    x = instance.x0
    eta = eta0
    start_time = time.time()
    offset_time = 0
    func_x = func(x)
    k = 0
    record_print_results()

    while k < maxiter:
        norm_F = jnp.sqrt(2 * func_x)
        mu_k = mu * norm_F ** delta
        eta = jnp.maximum(eta, mu_k)
        model_func, model_func_grad = model(x, mu_k)
        x_new, _, eta = apg_restart(
            model_func,
            model_func_grad,
            instance.proj,
            x,
            mu_k,
            eta,
            maxiter=maxiter_inner,
            tol_gradmap=tol_gradmap_inner,
            eta_inc=eta_inc,
            eta_dec=eta_dec,
        )
        func_new = func(x_new)

        # LM update
        if jnp.sqrt(2 * func_new) <= gamma * norm_F:
            x = x_new
            func_x = func_new
        # PG update
        else:
            grad_x = grad(x)
            alpha = 1
            while True:
                x_new = instance.proj(x - alpha * grad_x)
                func_new = func(x_new)
                if func(x_new) <= func_x + sigma * jnp.vdot(grad_x, x_new - x):
                    break
                else:
                    alpha *= beta
            x = x_new
            func_x = func_new
        k += 1

        record_print_results()
        if results[-1][0] > timeout or results[-1][2] <= tol_gradmap:
            break

    print(k, results[k])
    if return_sol:
        return results, sols
    else:
        return results


# [Facchinei (2013)] Algorithm 3
@mymodule.func_name
def facchinei2013_jvp(
    instance,
    gamma0=1,
    S=2,
    eta0=1e-0,
    eta_inc=2,
    eta_dec=0.9,
    tol_gradmap=0,
    tol_gradmap_inner=0,
    maxiter=100000,
    maxiter_inner=100000,
    timeout=100000,
    print_interval=10,
    return_sol=False,
):
    def func(x):
        return mymodule.sqnorm(instance.inner_func(x)) / 2

    def model(xk, lam):
        F, F_vjp = jax.vjp(instance.inner_func, xk)
        F_jvp = functools.partial(jax.jvp, instance.inner_func, (xk,))

        def func(x):
            r = F + F_jvp((x - xk,))[1]
            return (mymodule.sqnorm(r) + lam * mymodule.sqnorm(x - xk)) / 2

        def func_grad(x):
            r = F + F_jvp((x - xk,))[1]
            val = (mymodule.sqnorm(r) + lam * mymodule.sqnorm(x - xk)) / 2
            grad = F_vjp(r)[0] + lam * (x - xk)
            return val, grad

        return func, func_grad

    def record_print_results():
        nonlocal offset_time
        record_start = time.time()
        results.append(
            [
                time.time() - start_time - offset_time,
                func_x,
                norm_gradmap_fixedeta(x, func, instance.proj),
                eta,
            ]
        )
        if return_sol:
            sols.append(x)
        if print_interval is not None and k % print_interval == 0:
            print(k, results[-1][:])
            # print(x)
        offset_time += time.time() - record_start

    # initialization
    results = []
    sols = []
    x = instance.x0
    gamma = gamma0
    eta = eta0
    start_time = time.time()
    offset_time = 0
    func_x = func(x)
    k = 0
    record_print_results()

    while k < maxiter:
        norm_F = jnp.sqrt(2 * func_x)
        mu = norm_F ** 2
        eta = jnp.maximum(eta, mu)
        rhs = (gamma * mu) ** 2 / 2
        model_func, model_func_grad = model(x, mu)
        x, model_x_new, eta = apg_restart(
            model_func,
            model_func_grad,
            instance.proj,
            x,
            mu,
            eta,
            maxiter=maxiter_inner,
            tol_gradmap=tol_gradmap_inner,
            tol_f=rhs,
            eta_inc=eta_inc,
            eta_dec=eta_dec,
        )
        if model_x_new > rhs:
            gamma = S * jnp.sqrt(model_x_new * 2) / norm_F ** 2
        func_x = func(x)
        k += 1

        record_print_results()
        if results[-1][0] > timeout or results[-1][2] <= tol_gradmap:
            break

    print(k, results[k])
    if return_sol:
        return results, sols
    else:
        return results


# [Gonçalves et al. (2021)] Algorithm G-LMA-IP
@mymodule.func_name
def ggo2021(
    instance,
    M=1,
    eta1=1e-4,
    eta2=1e-2,
    eta3=1e10,
    beta=0.5,
    gamma=1e-3,
    tol_gradmap=0,
    maxiter=100000,
    timeout=100000,
    print_interval=10,
    return_sol=False,
):
    def func(x):
        return mymodule.sqnorm(instance.inner_func(x)) / 2

    grad = jax.grad(func)

    def record_print_results():
        nonlocal offset_time
        record_start = time.time()
        results.append(
            [
                time.time() - start_time - offset_time,
                func_x,
                norm_gradmap_fixedeta(x, func, instance.proj),
                alpha,
            ]
        )
        if return_sol:
            sols.append(x)
        if print_interval is not None and k % print_interval == 0:
            print(k, results[-1])
            # print(x)
        offset_time += time.time() - record_start

    # initialization
    results = []
    sols = []
    x = instance.x0
    start_time = time.time()
    offset_time = 0
    func_x = func(x)
    k = 0
    m = 0
    alpha = 1
    record_print_results()

    while k < maxiter:
        norm_F = jnp.sqrt(2 * func_x)
        mu_k = norm_F ** 2
        if instance.dim_range >= instance.dim_domain:
            J = jax.jacfwd(instance.inner_func)(x.reshape(-1))
        else:
            J = jax.jacrev(instance.inner_func)(x.reshape(-1))

        # Step 1
        grad_x = grad(x)
        R = jax.scipy.linalg.qr(
            jnp.concatenate((J, jnp.eye(instance.dim_domain) * jnp.sqrt(mu_k))),
            mode="r",
        )[: instance.dim_domain, :]
        tmp = jax.scipy.linalg.solve_triangular(R, -grad_x, trans="T")
        duk = jax.scipy.linalg.solve_triangular(R, tmp)

        # Step 2 and 3
        dk = instance.proj(x + duk) - x
        ip = grad_x @ dk
        norm_dk = scipy.linalg.norm(dk)
        norm_grad = scipy.linalg.norm(grad_x)
        if jnp.abs(ip) > norm_dk ** 2 * eta1 and eta2 <= norm_dk / norm_grad <= eta3:
            dk *= -jnp.sign(ip)
        else:
            dk = instance.proj(x - grad_x) - x

        # Step 4 and 5
        alpha = 1
        f_max = 0
        ip = grad_x @ dk
        for j in range(m + 1):
            f_max = jnp.maximum(f_max, results[k - j][1])
        while True:
            x_new = x + alpha * dk
            func_new = func(x_new)
            if func_new > f_max + gamma * alpha * ip:
                alpha *= beta
            else:
                break
        x = x_new
        func_x = func_new
        # print(k, func_x.item(), alpha)
        k += 1
        m = min(m + 1, M, k)

        record_print_results()
        if results[-1][0] > timeout or results[-1][2] <= tol_gradmap:
            break

    print(k, results[k])
    if return_sol:
        return results, sols
    else:
        return results

import numpy as np


def _heston_paths_full_truncation(
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
):
    """
    Simulate Heston under risk-neutral measure using full truncation Euler.
    dS/S = (r-q) dt + sqrt(v) dW1
    dv  = kappa(theta - v) dt + sigma sqrt(v) dW2
    corr(dW1,dW2)=rho
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)

    # correlated normals
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    w1 = z1
    w2 = rho * z1 + np.sqrt(max(1e-12, 1.0 - rho * rho)) * z2

    s = np.full(n_paths, s0, dtype=float)
    v = np.full(n_paths, v0, dtype=float)

    for i in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        # variance
        v = v + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * sqrt_dt * w2[:, i]
        v_pos_next = np.maximum(v, 0.0)

        # log-Euler for spot (stability)
        s = s * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * w1[:, i])

        # keep v as is (full trunc uses v_pos in drift/diffusion)
        v = v_pos_next

    return s


def price_european_call_mc(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    params: dict,
    n_steps: int = 200,
    n_paths: int = 50_000,
    seed: int = 42,
):
    """
    Monte Carlo price of a European call under Heston (synthetic/public).
    Returns: price, std_error
    """
    sT = _heston_paths_full_truncation(
        s0=s0,
        v0=params["v0"],
        r=r,
        q=q,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma=params["sigma"],
        rho=params["rho"],
        t=t,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    payoff = np.maximum(sT - k, 0.0)
    disc = np.exp(-r * t)
    price = disc * payoff.mean()
    std_error = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(std_error)


def convergence_table(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    params: dict,
    steps_list=(50, 100, 200),
    paths_list=(10_000, 30_000, 80_000),
    seed: int = 42,
):
    """
    Produce a small convergence grid for recruiters: (steps, paths) -> price ± SE.
    """
    out = []
    base_seed = seed
    for n_steps in steps_list:
        for n_paths in paths_list:
            price, se = price_european_call_mc(
                s0=s0, k=k, t=t, r=r, q=q, params=params,
                n_steps=n_steps, n_paths=n_paths, seed=base_seed
            )
            out.append((n_steps, n_paths, price, se))
            base_seed += 1
    return out

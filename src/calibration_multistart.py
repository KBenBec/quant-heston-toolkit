import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.heston_mc import price_european_call_mc


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _pack_params(x):
    return {"kappa": x[0], "theta": x[1], "sigma": x[2], "rho": x[3], "v0": x[4]}


def calibrate_heston_multistart(
    quotes: pd.DataFrame,
    s0: float,
    r: float,
    q: float,
    n_steps: int = 120,
    n_paths: int = 25_000,
    n_starts: int = 8,
    seed: int = 123,
):
    """
    Calibrate Heston parameters to synthetic option prices.
    quotes columns: T, K, price
    Objective: RMSE between MC prices and market prices.
    Multi-start bounded optimization to reduce local minima risk.
    """
    # bounds (reasonable / public-safe)
    bounds = [
        (0.2, 8.0),     # kappa
        (1e-4, 0.50),   # theta
        (0.05, 2.0),    # sigma
        (-0.95, -0.05), # rho (often negative in equity; adjust if you want)
        (1e-4, 0.50),   # v0
    ]

    rng = np.random.default_rng(seed)

    T = quotes["T"].to_numpy(float)
    K = quotes["K"].to_numpy(float)
    mkt = quotes["price"].to_numpy(float)

    def objective(x):
        params = _pack_params(x)
        model_prices = []
        # small speed trick: deterministic seeds per option
        local_seed = 1000
        for ti, ki in zip(T, K):
            p, _ = price_european_call_mc(
                s0=s0, k=float(ki), t=float(ti), r=r, q=q, params=params,
                n_steps=n_steps, n_paths=n_paths, seed=local_seed
            )
            model_prices.append(p)
            local_seed += 1
        model = np.array(model_prices, dtype=float)
        return rmse(model, mkt)

    best = None
    best_val = float("inf")

    # random starts within bounds
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best = res

    params_star = _pack_params(best.x)
    return params_star, best_val

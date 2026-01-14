import numpy as np
import pandas as pd
from src.heston_mc import price_european_call_mc


def greeks_fd(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    params: dict,
    n_steps: int = 160,
    n_paths: int = 40_000,
):
    """
    Finite-difference Greeks for a call:
    Delta, Gamma via spot bump; Vega via v0 bump.
    Uses common random numbers for stability.
    """
    eps_s = 0.01 * s0
    eps_v = 1e-3

    # common seed for CRN
    seed0 = 777

    p0, _ = price_european_call_mc(s0, k, t, r, q, params, n_steps, n_paths, seed0)

    # spot bumps
    p_up, _ = price_european_call_mc(s0 + eps_s, k, t, r, q, params, n_steps, n_paths, seed0)
    p_dn, _ = price_european_call_mc(s0 - eps_s, k, t, r, q, params, n_steps, n_paths, seed0)

    delta = (p_up - p_dn) / (2.0 * eps_s)
    gamma = (p_up - 2.0 * p0 + p_dn) / (eps_s ** 2)

    # vega via v0 bump
    params_up = dict(params)
    params_dn = dict(params)
    params_up["v0"] = max(1e-8, params["v0"] + eps_v)
    params_dn["v0"] = max(1e-8, params["v0"] - eps_v)

    pv_up, _ = price_european_call_mc(s0, k, t, r, q, params_up, n_steps, n_paths, seed0)
    pv_dn, _ = price_european_call_mc(s0, k, t, r, q, params_dn, n_steps, n_paths, seed0)
    vega = (pv_up - pv_dn) / (2.0 * eps_v)

    return {"price": p0, "delta": float(delta), "gamma": float(gamma), "vega_v0": float(vega)}


def stress_grid(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    params: dict,
    shocks_v0=(-0.02, 0.0, 0.02),
    shocks_r=(-0.005, 0.0, 0.005),
    n_steps: int = 140,
    n_paths: int = 35_000,
):
    """
    Simple stress grid on v0 and r; exports a dataframe.
    """
    rows = []
    base_seed = 900
    for dv0 in shocks_v0:
        for dr in shocks_r:
            p = dict(params)
            p["v0"] = max(1e-8, params["v0"] + dv0)
            pr, _ = price_european_call_mc(s0, k, t, r + dr, q, p, n_steps, n_paths, base_seed)
            rows.append({"dv0": dv0, "dr": dr, "price": float(pr)})
            base_seed += 1
    return pd.DataFrame(rows)

import pandas as pd
from src.heston_mc import price_european_call_mc
from src.calibration_multistart import calibrate_heston_multistart


def make_synthetic_quotes():
    true_params = {"kappa": 1.8, "theta": 0.05, "sigma": 0.55, "rho": -0.65, "v0": 0.05}
    s0, r, q = 100.0, 0.02, 0.0
    grid = [(0.5, 90.0), (0.5, 100.0), (0.5, 110.0),
            (1.0, 90.0), (1.0, 100.0), (1.0, 110.0)]
    rows = []
    seed = 2000
    for T, K in grid:
        price, _ = price_european_call_mc(s0, K, T, r, q, true_params, n_steps=180, n_paths=40_000, seed=seed)
        rows.append({"T": T, "K": K, "price": price})
        seed += 1
    return pd.DataFrame(rows), true_params


if __name__ == "__main__":
    quotes, true_params = make_synthetic_quotes()
    s0, r, q = 100.0, 0.02, 0.0

    est_params, fit_rmse = calibrate_heston_multistart(
        quotes=quotes,
        s0=s0, r=r, q=q,
        n_steps=120, n_paths=25_000,
        n_starts=8
    )

    print("True params:", true_params)
    print("Est. params:", est_params)
    print(f"Fit RMSE: {fit_rmse:.6f}")

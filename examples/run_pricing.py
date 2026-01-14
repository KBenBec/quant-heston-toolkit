from src.heston_mc import price_european_call_mc, convergence_table

if __name__ == "__main__":
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.6, "rho": -0.7, "v0": 0.04}
    s0, k, t, r, q = 100.0, 100.0, 1.0, 0.02, 0.0

    price, se = price_european_call_mc(s0, k, t, r, q, params, n_steps=200, n_paths=60_000)
    print(f"MC price: {price:.4f}  (SE: {se:.4f})")

    table = convergence_table(s0, k, t, r, q, params)
    print("\nConvergence grid (steps, paths, price, SE):")
    for row in table:
        print(row)

from src.greeks_stress import greeks_fd, stress_grid

if __name__ == "__main__":
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.6, "rho": -0.7, "v0": 0.04}
    s0, k, t, r, q = 100.0, 100.0, 1.0, 0.02, 0.0

    g = greeks_fd(s0, k, t, r, q, params)
    print("Greeks (FD):", g)

    df = stress_grid(s0, k, t, r, q, params)
    print("\nStress grid:")
    print(df)

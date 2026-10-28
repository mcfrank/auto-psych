# name: far_target_yes_rate
# description: Yes rate to targets more than 20 away from the nearest example, on sets of >=2 examples; observed above null means the model generalizes too little far from the examples (too range-bound), below null means it generalizes too much.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (n >= 2) & (d > 20)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())

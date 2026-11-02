# name: mixed_parity_interior_gap_yes_rate
# description: Yes rate for targets strictly inside the range of the examples but more than 8 from every example, among sets of >=3 examples with mixed parity (no parity rule); t_observed below null_mean means people fill gaps between examples less than the model's interval hypotheses predict (similarity-based rather than range-based), above means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    mixed = np.nanmin(par, axis=1) != np.nanmax(par, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    inside = (t > np.nanmin(ex, axis=1)) & (t < np.nanmax(ex, axis=1))
    m = mixed & (n >= 3) & inside & (d > 8)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())

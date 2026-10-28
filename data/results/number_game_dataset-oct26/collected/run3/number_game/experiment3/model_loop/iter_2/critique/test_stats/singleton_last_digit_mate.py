# name: singleton_last_digit_mate
# description: For one-example sets, yes-rate for targets that share the example's last digit and lie more than 10 away from it; observed > null means a single example already triggers last-digit rule generalization more strongly than the model's weak single-example evidence plus guessing predicts (observed < null: weaker).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    e = E[:, 0]
    m = (n == 1) & (np.mod(t, 10) == np.mod(e, 10)) & (np.abs(t - e) > 10)
    return float(y[m].mean()) if m.any() else 0.0

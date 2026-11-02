# name: single_shared_final_digit_yes
# description: Yes rate for targets whose final digit matches exactly one example's final digit (sets of 2+ examples, target not itself an example); observed below null means the model over-weights a single shared last digit (ends-in rules) relative to people.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    fd = np.nansum((X % 10) == (t % 10)[:, None], axis=1)
    m = (n >= 2) & (fd == 1) & (dist > 0)
    return float(y[m].mean()) if m.sum() > 0 else 0.0

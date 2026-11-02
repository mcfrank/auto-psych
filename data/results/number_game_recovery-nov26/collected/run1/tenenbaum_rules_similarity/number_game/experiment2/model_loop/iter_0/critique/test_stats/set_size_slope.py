# name: set_size_slope
# description: OLS slope of yes rate (excluding targets that are examples) on the number of examples (1-4); observed less negative than null means people tighten generalisation with more examples less than the size principle predicts, more negative means they tighten more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1).astype(float)
    t = df["target"].astype(float).to_numpy()
    m = ~(ex == t[:, None]).any(axis=1)
    x = n[m]; y = df["rating"].astype(float).to_numpy()[m]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])

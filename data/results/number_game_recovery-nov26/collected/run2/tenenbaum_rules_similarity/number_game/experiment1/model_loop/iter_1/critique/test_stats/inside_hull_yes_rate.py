# name: inside_hull_yes_rate
# description: Yes-rate for targets strictly between the smallest and largest example (not themselves examples); observed > null means people fill in the example range more than the model, observed < null means the model over-predicts interval-style "yes" inside the range.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    mask = (t > lo) & (t < hi) & ~member
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(float)[mask].mean())

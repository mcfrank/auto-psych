# name: last_vs_first_example_proximity
# description: Yes-rate for targets closer to the last-shown example than to the first minus yes-rate for targets closer to the first (multi-example sets); the model is order-invariant, so observed > null means a recency effect of example order, observed < null a primacy effect.
def test_statistic(df):
    parts = df["number_set"].str.split(",")
    first = pd.to_numeric(parts.str[0]).to_numpy(float)
    last = pd.to_numeric(parts.str[-1]).to_numpy(float)
    t = df["target"].astype(float).to_numpy()
    n = df["number_set"].str.count(",").to_numpy() + 1
    dl = np.abs(t - last); dfst = np.abs(t - first)
    r = df["rating"].to_numpy(float)
    a = (n >= 2) & (dl < dfst); b = (n >= 2) & (dfst < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

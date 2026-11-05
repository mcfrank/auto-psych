# name: spread_endorsement_slope
# description: Correlation across multi-example sets between the set's range (max-min) and its overall yes-rate; observed > null means wide-spread sets boost generalization more than the model's variability-tuned attention predicts, observed < null means less.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return ex, t, y

def test_statistic(df):
    ex, t, y = _parse(df)
    n = (~np.isnan(ex)).sum(1)
    rng = np.nanmax(ex, 1) - np.nanmin(ex, 1)
    m = n >= 2
    s = pd.DataFrame({"s": df["number_set"].to_numpy()[m], "r": rng[m], "y": y[m]})
    g = s.groupby("s").agg(r=("r", "first"), y=("y", "mean"))
    if len(g) < 3 or g["r"].std() == 0 or g["y"].std() == 0:
        return 0.0
    return float(np.corrcoef(g["r"], g["y"])[0, 1])

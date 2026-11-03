# name: single_example_distance_slope
# description: Among one-example sets, Pearson correlation between yes and log(1+|target-example|); more negative observed than null means people show stronger numerical-magnitude similarity gradients than the model, less negative means weaker.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",").to_numpy()
    if one.sum() < 3:
        return 0.0
    x = pd.to_numeric(s[one], errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()[one]
    y = df["rating"].astype(float).to_numpy()[one]
    d = np.log1p(np.abs(t - x))
    ok = np.isfinite(d)
    d, y = d[ok], y[ok]
    if d.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(d, y)[0, 1])

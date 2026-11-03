# name: near_example_yes_rate
# description: Yes rate for non-example targets within distance 2 of the nearest example number (all sets); observed > null means people show more local numeric-similarity generalization than the model, observed < null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    vals = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    dist = np.nanmin(np.abs(vals - t[:, None]), axis=1)
    m = (dist >= 1) & (dist <= 2)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())

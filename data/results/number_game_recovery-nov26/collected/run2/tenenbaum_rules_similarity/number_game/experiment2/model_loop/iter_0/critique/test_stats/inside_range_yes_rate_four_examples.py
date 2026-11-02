# name: inside_range_yes_rate_four_examples
# description: Yes rate for non-example targets lying strictly between the min and max of a four-example set; observed below the null means the model over-generalizes to the spanned interval on rich sets (above = under-generalizes).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).values
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    notin = ~(ex == t[:, None]).any(axis=1)
    m = (n == 4) & (t > lo) & (t < hi) & notin
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

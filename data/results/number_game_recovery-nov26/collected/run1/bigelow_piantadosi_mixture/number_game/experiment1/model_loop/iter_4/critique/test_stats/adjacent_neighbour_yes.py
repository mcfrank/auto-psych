# name: adjacent_neighbour_yes
# description: Yes-rate for non-example targets 1-3 away from some example in multi-example sets; observed < null means people endorse immediate numerical neighbours of examples less than the model (whose range guesses nearly always cover them), observed > null means more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    P = parts.values
    n = parts.notna().sum(axis=1).values
    t = df["target"].astype(float).values
    d = np.nanmin(np.abs(P - t[:, None]), axis=1)
    m = (n >= 2) & (d >= 1) & (d <= 3)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

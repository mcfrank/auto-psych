# name: zero_yes_cell_fraction
# description: Fraction of non-example (set,target) cells whose responses are all "no"; observed above null means people give categorical rejections more often than the model (model too graded/probabilistic for clear non-members); below = model produces too many unanimous rejections.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    d = pd.DataFrame({"s": df["number_set"].values, "t": t, "r": r})[~isex]
    if len(d) == 0: return 0.0
    c = d.groupby(["s", "t"])["r"].max()
    return float((c == 0).mean())

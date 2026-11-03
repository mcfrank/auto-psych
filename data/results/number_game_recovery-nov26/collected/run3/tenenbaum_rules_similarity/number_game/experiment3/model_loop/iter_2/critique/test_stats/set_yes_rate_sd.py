# name: set_yes_rate_sd
# description: SD across distinct number sets (with >=20 trials) of the set-level yes rate on non-example targets; observed above null means people's answers differ more between sets (more extreme acceptance/rejection per set) than the model's predictions, below means the model is too extreme across sets.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",")
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    mat = np.full((len(df), 4), -1)
    for j in range(4):
        mat[:, j] = pd.to_numeric(s.str[j], errors="coerce").fillna(-1).astype(int).values
    valid = mat > 0
    n = valid.sum(1)
    shown = np.any(valid & (mat == t[:, None]), 1)
    lo = np.where(valid, mat, 999).min(1)
    hi = np.where(valid, mat, -1).max(1)
    key = df["number_set"].astype(str).values
    m = ~shown
    g = pd.DataFrame({"k": key[m], "r": r[m]}).groupby("k")["r"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    return float(g["mean"].std()) if len(g) > 1 else 0.0

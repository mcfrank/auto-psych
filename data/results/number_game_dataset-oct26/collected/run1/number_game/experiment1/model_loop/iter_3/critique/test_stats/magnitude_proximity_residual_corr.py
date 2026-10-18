# name: magnitude_proximity_residual_corr
# description: Correlation between rating and -log(1 + distance from target to nearest example), after demeaning both within strata of identical final-digit/parity feature values (the model's only inputs); observed above null (~0) means people credit numerical closeness that the digit-only model ignores, below means they penalize it.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    same_dig = np.nansum((ex % 10) == (t[:, None] % 10), axis=1) / n
    same_par = np.nansum(((ex % 2) == (t[:, None] % 2)) & ((ex % 10) != (t[:, None] % 10)), axis=1) / n
    key = pd.Series(np.round(same_dig, 3).astype(str)) + "_" + pd.Series(np.round(same_par, 3).astype(str))
    prox = -np.log1p(d)
    r = df["rating"].to_numpy().astype(float)
    tmp = pd.DataFrame({"k": key.values, "p": prox, "r": r})
    pr = tmp["p"] - tmp.groupby("k")["p"].transform("mean")
    rr = tmp["r"] - tmp.groupby("k")["r"].transform("mean")
    den = np.sqrt((pr ** 2).sum() * (rr ** 2).sum())
    return float((pr * rr).sum() / den) if den > 0 else 0.0

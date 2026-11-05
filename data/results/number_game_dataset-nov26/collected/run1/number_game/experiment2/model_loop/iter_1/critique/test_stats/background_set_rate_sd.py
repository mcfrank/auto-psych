# name: background_set_rate_sd
# description: Standard deviation across sets (with >=9 'background' trials: targets sharing no example's final digit and >=10 from every example) of each set's background yes-rate; observed > null means sets differ in how broadly people generalize to dissimilar numbers (set-specific rule breadth, e.g. "4,66,78,96" vs "80,70,90,100") more than the model's similarity features produce, observed < null means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    M = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        l = [int(v) for v in s.split(",")][:4]
        M[i, :len(l)] = l
    idx = pd.Series(np.arange(len(uniq)), index=uniq)[sets].to_numpy()
    E = M[idx]
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    D = np.where(np.isnan(E), np.inf, np.abs(E - t[:, None]))
    dm = ((np.nan_to_num(E, nan=-1) % 10) == (t % 10)[:, None]).any(1)
    bg = (~dm) & (D.min(1) >= 10)
    g = pd.DataFrame({"s": idx[bg], "r": r[bg]}).groupby("s")["r"].agg(["mean", "size"])
    g = g[g["size"] >= 9]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())

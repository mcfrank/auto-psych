# name: single_set_breadth_dispersion
# description: Standard deviation across single-example sets (with >=20 responses) of the set-level yes-rate on non-example targets; observed above null means people's generalization breadth varies more from seed to seed (some single numbers suggest broad, others narrow concepts) than the model produces, below means the model makes seeds too different.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    isex = np.any(E == t[:, None], axis=1)
    ok = (n == 1) & ~isex
    s = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy()[ok], "r": r[ok]})
    g = s.groupby("s")["r"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())

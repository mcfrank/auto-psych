# name: pow2_vs_other_even_contrast
# description: For sets whose examples are all powers of two (e.g. 16,32,2,8; 2,8; 8,64; 64,4), yes rate on non-example powers of two minus yes rate on other even targets; observed above null_mean means people restrict to the small nested rule more sharply than the model (model over-generalizes to the larger "even"/multiples rules), below means the model is too sharp.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pow2 = np.array([2, 4, 8, 16, 32, 64], dtype=float)
    isp = np.isin(ex, pow2) | np.isnan(ex)
    all_p = isp.all(axis=1) & ~np.isnan(ex).all(axis=1)
    t = df["target"].astype(int).to_numpy()
    is_ex = (ex == t[:, None]).any(axis=1)
    r = df["rating"].to_numpy().astype(float)
    tp = np.isin(t, pow2)
    a = all_p & ~is_ex & tp
    b = all_p & ~is_ex & ~tp & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

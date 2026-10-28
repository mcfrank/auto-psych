# name: shared_divisor_gap_multi
# description: On sets of 2+ examples whose gcd is >= 3, yes-rate for (non-example) targets divisible by that gcd minus yes-rate for targets not divisible; observed above null means people use the arithmetic (multiples-of) rule more than the model predicts, below means the model over-weights the rule relative to similarity.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = parts.notna().sum(axis=1).to_numpy()
    arr = parts.fillna(0).to_numpy().astype(np.int64)
    g = np.gcd.reduce(arr, axis=1)
    tg = df["target"].to_numpy().astype(np.int64)
    inset = (arr == tg[:, None]).any(axis=1)
    base = (n >= 2) & (g >= 3) & ~inset
    div = base & (tg % np.maximum(g, 1) == 0)
    nd = base & (tg % np.maximum(g, 1) != 0)
    if not div.any() or not nd.any():
        return 0.0
    y = df["rating"].to_numpy(float)
    return float(y[div].mean() - y[nd].mean())

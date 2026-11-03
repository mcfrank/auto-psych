# name: single_example_same_decade_contrast
# description: For single-example sets, yes rate for targets in the same decade (same tens digit, 100 counted with 91-99) as the example minus yes rate for targets in other decades; observed above null means people generalise within the example's decade more than the model predicts, below means the model over-produces decade/proximity generalisation.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").to_numpy()
    y = df["rating"].astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    e = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).astype(int).to_numpy()
    dec = lambda x: np.minimum((x - 1) // 10, 9)
    same = dec(t) == dec(np.maximum(e, 1))
    a, b = single & same, single & ~same
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())

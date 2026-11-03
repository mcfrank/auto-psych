# name: adjacent_to_example_yes
# description: Yes rate for targets exactly 1 or 2 away from some example (not themselves examples), all sets; observed above the null means the model under-produces near-neighbour (numerical similarity) generalization, below means it over-produces it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (dist >= 1) & (dist <= 2)
    return float(y[m].mean()) if m.any() else 0.0

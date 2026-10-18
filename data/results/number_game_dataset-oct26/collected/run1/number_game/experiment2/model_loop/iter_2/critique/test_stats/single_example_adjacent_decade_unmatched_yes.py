# name: single_example_adjacent_decade_unmatched_yes
# description: In single-example sets, yes rate for targets one decade away from the example that share neither its final digit nor its parity; t_observed above null_mean means the model under-produces generalization from one example to nearby numbers of opposite parity (a broad-magnitude hypothesis), below means it over-produces it.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",").values
    t = df["target"].astype(int).values
    x = pd.to_numeric(sets.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    sel = single & (np.abs(t // 10 - x // 10) == 1) & ((t % 2) != (x % 2))
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[sel].mean())

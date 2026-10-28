# name: even_target_yes_after_all_odd_sets
# description: Yes rate to even targets on sets of >=2 examples that are all odd; observed above null means the model under-produces "yes" to parity-violating targets (people are less sure of the "odd" rule / noisier than the model), below null means it over-produces them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    all_odd = np.all(np.isnan(ex) | (np.mod(ex, 2) == 1), axis=1)
    t = df["target"].astype(int).to_numpy()
    m = (n >= 2) & all_odd & (t % 2 == 0)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())

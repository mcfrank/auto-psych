# name: odd_set_odd_target_yes
# description: Yes rate for odd targets (not themselves examples) when all examples are odd and there are >=2 of them; observed below the null means the model over-produces "odd numbers" generalization (gives the parity rule too much weight), above means it under-produces it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    allodd = np.all(~valid | (np.nan_to_num(ex) % 2 == 1), axis=1)
    inset = np.any(ex == t[:, None], axis=1)
    m = allodd & (n >= 2) & (t % 2 == 1) & ~inset
    return float(y[m].mean()) if m.any() else 0.0

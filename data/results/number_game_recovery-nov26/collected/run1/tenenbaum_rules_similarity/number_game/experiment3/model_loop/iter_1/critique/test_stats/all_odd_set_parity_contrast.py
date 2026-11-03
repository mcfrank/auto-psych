# name: all_odd_set_parity_contrast
# description: Among multi-example sets whose examples are all odd, yes rate for odd targets minus yes rate for even targets (targets not in the set); observed below null means people generalise parity from all-odd sets less than the model predicts (model over-weights the 'odd' rule), above means more.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    table = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        table[i, :len(v)] = v
    ex = table[codes]
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    allodd = np.all(np.isnan(ex) | (np.mod(ex, 2) == 1), axis=1) & (n > 1)
    notin = ~np.any(ex == t[:, None], axis=1)
    a = allodd & notin & (np.mod(t, 2) == 1)
    b = allodd & notin & (np.mod(t, 2) == 0)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())

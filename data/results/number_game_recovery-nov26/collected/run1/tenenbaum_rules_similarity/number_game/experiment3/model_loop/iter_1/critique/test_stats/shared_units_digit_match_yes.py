# name: shared_units_digit_match_yes
# description: Among sets of 2+ examples that all end in the same units digit, yes rate for non-example targets ending in that digit; observed below null means people extend 'ends in d' rules less than the model predicts, above means more.
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
    u = np.mod(ex, 10)
    u0 = u[:, 0]
    same = np.all(np.isnan(ex) | (u == u0[:, None]), axis=1) & (n > 1)
    notin = ~np.any(ex == t[:, None], axis=1)
    m = same & notin & (np.mod(t, 10) == u0)
    return float(y[m].mean()) if m.any() else 0.0

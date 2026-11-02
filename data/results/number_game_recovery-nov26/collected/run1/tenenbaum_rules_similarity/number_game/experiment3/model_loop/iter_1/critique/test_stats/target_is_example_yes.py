# name: target_is_example_yes
# description: Yes rate on trials where the target is itself one of the set's examples; observed below null means people endorse shown examples less than the model (which treats examples as near-certain members) predicts, above means the model under-endorses them.
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
    m = np.any(ex == t[:, None], axis=1)
    return float(y[m].mean()) if m.any() else 0.0

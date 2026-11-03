# name: all_even_set_odd_target_yes
# description: Among multi-example sets whose examples are all even, yes rate for odd targets; observed above null means people break parity (endorse odd numbers after even examples) more than the model predicts, below means the model over-predicts parity violations.
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
    alleven = np.all(np.isnan(ex) | (np.mod(ex, 2) == 0), axis=1) & (n > 1)
    m = alleven & (np.mod(t, 2) == 1)
    return float(y[m].mean()) if m.any() else 0.0

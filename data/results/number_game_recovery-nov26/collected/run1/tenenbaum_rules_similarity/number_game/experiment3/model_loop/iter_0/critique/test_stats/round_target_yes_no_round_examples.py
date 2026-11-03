# name: round_target_yes_no_round_examples
# description: Yes rate for targets that are multiples of 10, among sets containing no multiple of 10; observed above null means people favour round-number targets (a round-number bias the model lacks), below means the model over-endorses them.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    parsed = {s: [int(x) for x in s.split(",")] for s in uniq}
    ex = np.full((len(df), 4), np.nan)
    codes = pd.Categorical(sets, categories=uniq).codes
    table = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = parsed[s][:4]
        table[i, :len(v)] = v
    ex = table[codes]
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    any_round = np.any(np.mod(ex, 10) == 0, axis=1)
    m = ~any_round & (np.mod(t, 10) == 0)
    return float(y[m].mean()) if m.any() else 0.0

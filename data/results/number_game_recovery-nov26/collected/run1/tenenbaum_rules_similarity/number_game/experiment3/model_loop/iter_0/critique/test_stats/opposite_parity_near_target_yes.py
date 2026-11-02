# name: opposite_parity_near_target_yes
# description: Yes rate for targets of the opposite parity to the examples lying within 5 of some example, among sets of 2+ examples that all share one parity; observed above null means people generalise by numeric proximity even against a parity regularity more than the model predicts (model too rule-bound), below means the model over-generalises across parity.
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
    p = np.mod(ex, 2)
    same_par = (np.nanmax(p, axis=1) == np.nanmin(p, axis=1))
    opp = np.mod(t, 2) != np.nanmax(p, axis=1)
    near = np.nanmin(np.abs(ex - t[:, None]), axis=1) <= 5
    m = (n >= 2) & same_par & opp & near
    return float(y[m].mean()) if m.any() else 0.0

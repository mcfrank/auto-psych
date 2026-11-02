# name: mixed_ending_units_digit_match_yes
# description: Yes rate for targets whose units digit matches at least one example's units digit (target not itself an example), among multi-example sets whose examples do NOT all share one units digit; observed above null means people generalise by digit similarity that the model's single-digit rules cannot express (model under-produces), below means the reverse.
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
    u = np.mod(ex, 10)
    same_end = np.nanmax(u, axis=1) == np.nanmin(u, axis=1)
    match = np.any(u == np.mod(t, 10)[:, None], axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    m = (n >= 2) & ~same_end & match & ~is_ex
    return float(y[m].mean()) if m.any() else 0.0

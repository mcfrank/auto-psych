# name: even_set_odd_target_yes_rate
# description: Yes rate for odd targets after sets whose examples are all even (2+ examples); observed > null means the model over-commits to parity/even rules and people accept parity-violating numbers more, observed < null means people respect parity more strictly than the model.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ex = s.str.split(",", expand=True)
    n_ex = (s.str.count(",") + 1).values
    all_even = np.ones(len(df), dtype=bool)
    for c in ex.columns:
        col = ex[c]
        v = col.fillna("0").astype(int).values
        all_even &= (col.isna().values) | (v % 2 == 0)
    m = all_even & (n_ex >= 2) & (t % 2 == 1)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())

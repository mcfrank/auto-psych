# name: odd_set_parity_contrast
# description: Among sets whose examples are all odd but do not share a last digit, yes-rate for odd targets minus yes-rate for even targets; observed < model means people generalise the "odd" concept less (or rely on other features) than the model's parity rule predicts; observed > model means stronger parity generalisation.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(sets), axis=1)
    allodd = np.all((np.mod(sets, 2) == 1) | np.isnan(sets), axis=1)
    ld = np.mod(sets, 10)
    same_ld = np.all((ld == ld[:, [0]]) | np.isnan(ld), axis=1)
    sel = allodd & ~same_ld & (n >= 2)
    todd = np.mod(df["target"].astype(int).to_numpy(), 2) == 1
    r = df["rating"].to_numpy().astype(float)
    a, b = sel & todd, sel & ~todd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

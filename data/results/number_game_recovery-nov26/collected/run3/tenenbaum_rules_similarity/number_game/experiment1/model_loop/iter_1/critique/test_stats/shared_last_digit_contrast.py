# name: shared_last_digit_contrast
# description: Among sets of >=2 examples that all share one last digit, yes-rate for targets with that last digit minus yes-rate for targets without it; observed < model means people's last-digit generalisation is less sharp (they accept non-matching or reject matching targets more) than the model's "ends in d" rule predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(sets), axis=1)
    ld = np.mod(sets, 10)
    first = ld[:, 0]
    same = np.all((ld == first[:, None]) | np.isnan(ld), axis=1) & (n >= 2)
    tld = np.mod(df["target"].astype(int).to_numpy(), 10)
    match = tld == first
    r = df["rating"].to_numpy().astype(float)
    a, b = same & match, same & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

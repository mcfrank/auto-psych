# name: single_example_last_digit_match
# description: Among sets whose examples do NOT all share a last digit, for non-example targets more than 5 away from every example, yes-rate for targets ending in the same digit as at least one example minus yes-rate for targets ending in no example's digit; observed above model means people generalise from partial surface similarity to single examples (last digit) beyond what the model's whole-set-consistent rules allow, below means less.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    ld = np.where(np.isnan(ex), np.nan, ex % 10)
    first = ld[:, 0]
    all_same = np.all(np.isnan(ld) | (ld == first[:, None]), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    match = np.any(ld == (t % 10)[:, None], axis=1)
    base = (~all_same) & (dist > 5)
    r = df["rating"].to_numpy()
    a, b = base & match, base & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

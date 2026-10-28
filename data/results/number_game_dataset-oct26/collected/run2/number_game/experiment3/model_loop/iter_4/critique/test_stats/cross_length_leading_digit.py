# name: cross_length_leading_digit
# description: Among sets whose examples are all one-digit, yes-rate on two-digit targets whose first digit equals some example's digit (35 after 3) minus two-digit targets whose first digit matches none; observed < null means people do not carry the leading-digit match across one- vs two-digit numbers as the model assumes (model over-produces this boost), observed > null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    onedig = (ex.max(axis=1) < 10).values
    lead = t.astype(str).str[0].astype(int).values
    match = np.zeros(len(df), dtype=bool)
    for c in ex.columns:
        v = ex[c].values
        match |= (v < 10) & (v == lead)
    sel = onedig & (t.values >= 10) & (t.values <= 99)
    r = df["rating"].values.astype(float)
    a = sel & match; b = sel & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

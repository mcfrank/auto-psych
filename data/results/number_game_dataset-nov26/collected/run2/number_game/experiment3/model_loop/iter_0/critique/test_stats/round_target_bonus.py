# name: round_target_bonus
# description: Among sets whose examples are not all multiples of 5, yes rate for targets that are multiples of 10 minus yes rate for other targets; observed > null means people favour round targets per se (a set-independent prior) which the model, crediting only shared properties, under-produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    allfive = ((ex % 5 == 0) | ex.isna()).all(axis=1)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    m = ~allfive
    a = r[m & (t % 10 == 0)]; b = r[m & (t % 10 != 0)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())

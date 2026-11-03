# name: gcd2_far_even_yes_rate
# description: For sets of 2+ all-even examples whose gcd is exactly 2 and not sharing a final digit, yes rate to even non-example targets more than 10 from every example; observed above null means the model under-predicts "even numbers" generalisation, below means it over-predicts it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").reindex(columns=range(4)).values
    ok = ~np.isnan(ex)
    exi = np.where(ok, ex, 0).astype(int)
    n = ok.sum(1)
    g = np.gcd.reduce(exi, axis=1)
    last = np.where(ok, exi % 10, -1)
    same_last = np.all((last == last[:, [0]]) | ~ok, axis=1)
    t = df["target"].astype(int).values
    dist = np.min(np.where(ok, np.abs(exi - t[:, None]), 1000), axis=1)
    m = (n >= 2) & (g == 2) & ~same_last & (t % 2 == 0) & (dist > 10)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

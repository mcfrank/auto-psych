# name: prime_set_extension_contrast
# description: In sets of >=2 examples that are all primes, yes rate on non-example prime targets minus yes rate on non-prime odd targets; observed > null means people generalise along primality more than the model, observed < null means the model over-weights the primes rule.
def test_statistic(df):
    primes = {n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1))}
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    sel = (ex.isin(primes) | ex.isna()).all(axis=1) & (ex.notna().sum(axis=1) >= 2)
    t = df["target"].astype(int)
    inset = ex.eq(t, axis=0).any(axis=1)
    pr = sel & t.isin(primes) & ~inset
    other = sel & ~t.isin(primes) & (t % 2 == 1)
    if pr.sum() == 0 or other.sum() == 0:
        return 0.0
    return float(df.loc[pr, "rating"].mean() - df.loc[other, "rating"].mean())

# name: even_vs_odd_set_parity_asymmetry
# description: (Yes-rate on even minus odd non-example targets in all-even sets of 2+ examples) minus (yes-rate on odd minus even non-example targets in all-odd sets of 2+ examples); observed > null means people abstract an "even numbers" rule more than an "odd numbers" rule, an asymmetry the model's symmetric parity similarity cannot produce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).values
    k = ex.notna().sum(axis=1).values
    par = ex % 2
    alleven = ((par == 0) | ex.isna()).all(axis=1).values & (k >= 2)
    allodd = ((par == 1) | ex.isna()).all(axis=1).values & (k >= 2)
    isex = (ex.values == t[:, None]).any(axis=1)
    r = df["rating"].values.astype(float)
    ev = t % 2 == 0
    def m(x):
        return r[x].mean() if x.sum() > 0 else np.nan
    de = m(alleven & ~isex & ev) - m(alleven & ~isex & ~ev)
    do = m(allodd & ~isex & ~ev) - m(allodd & ~isex & ev)
    v = de - do
    return float(v) if np.isfinite(v) else 0.0

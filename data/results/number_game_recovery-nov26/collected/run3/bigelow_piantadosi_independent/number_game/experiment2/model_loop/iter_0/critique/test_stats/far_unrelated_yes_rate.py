# name: far_unrelated_yes_rate
# description: Yes rate on targets more than 15 from every example, sharing no last digit with any example, and of the opposite parity to an all-even or all-odd set; observed above null means the model under-produces 'scattered guess' yeses to unrelated numbers (its baseline is too low), and below null means it over-produces them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    mind = ex.sub(t, axis=0).abs().min(axis=1)
    par = ex % 2
    alleven = (par.fillna(0) == 0).all(axis=1)
    allodd = (par.fillna(1) == 1).all(axis=1)
    mism = (alleven & (t % 2 == 1)) | (allodd & (t % 2 == 0))
    samel = (ex % 10).eq(t % 10, axis=0).any(axis=1)
    m = ((mind > 15) & ~samel & mism).values
    return float(r[m].mean()) if m.sum() > 0 else 0.0

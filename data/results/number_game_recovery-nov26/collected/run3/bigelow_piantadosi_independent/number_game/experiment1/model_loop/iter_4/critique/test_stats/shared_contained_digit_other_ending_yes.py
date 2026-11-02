# name: shared_contained_digit_other_ending_yes
# description: For 2+ example sets in which every example contains some digit d, yes rate on non-example targets that contain such a d but whose last digit matches no example's last digit; observed > null means people use "contains digit d" more than the model predicts, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    inset = ex.eq(t, axis=0).any(axis=1)
    tt, to = (t // 10), (t % 10)
    hit = pd.Series(False, index=df.index)
    for dg in range(10):
        has = ((ex % 10 == dg) | ((ex // 10 == dg) & (ex >= 10) & (ex < 100)) | ((dg == 1) & (ex == 100)) | ((dg == 0) & (ex == 100)) | ex.isna()).all(axis=1)
        thas = (to == dg) | ((tt == dg) & (t >= 10) & (t < 100)) | ((t == 100) & ((dg == 1) | (dg == 0)))
        hit = hit | (has & thas)
    lastmatch = (ex % 10).eq(to, axis=0).any(axis=1)
    sel = hit & (n >= 2) & ~inset & ~lastmatch
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[sel].mean())

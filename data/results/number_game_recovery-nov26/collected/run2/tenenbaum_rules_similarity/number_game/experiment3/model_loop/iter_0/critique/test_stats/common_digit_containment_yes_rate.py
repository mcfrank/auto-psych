# name: common_digit_containment_yes_rate
# description: For sets of 2+ examples in which some digit appears in every example, yes rate on non-example targets that contain that digit but do not share any example's final digit; observed above null means people generalise "contains digit d" more than the model (which has no such concepts), below means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(int).values
    n = (~np.isnan(ex)).sum(axis=1)
    exi = np.nan_to_num(ex, nan=-1).astype(int)
    valid = exi >= 0
    tens, ones = exi // 10, exi % 10
    hit = np.zeros(len(df), dtype=bool)
    for d in range(10):
        has = ((ones == d) | ((tens % 10 == d) & (exi >= 10)) | ((exi == 100) & (d in (0, 1)))) | ~valid
        common = has.all(axis=1) & (n >= 2)
        tcont = (t % 10 == d) | ((t // 10 % 10 == d) & (t >= 10)) | ((t == 100) & (d in (0, 1)))
        hit |= common & tcont
    same_final = ((ones == (t % 10)[:, None]) & valid).any(axis=1)
    in_set = (exi == t[:, None]).any(axis=1)
    m = hit & ~same_final & ~in_set
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

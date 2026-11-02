# name: far_shared_last_digit_yes
# description: Among sets of >=2 examples that all share one last digit, yes rate for non-example targets with that last digit lying >=25 from the nearest example; observed > null means the model's distance hesitance over-discounts far members of a digit rule (people extend "ends in d" across the whole range), observed < null that it under-discounts them.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).fillna(0).astype(int).values
    has = ex > 0
    nex = has.sum(1)
    t = df["target"].astype(int).values
    ld = np.where(has, ex % 10, -1)
    first = ld[:, 0]
    same = ((ld == first[:, None]) | ~has).all(1)
    dist = np.where(has, np.abs(ex - t[:, None]), 999).min(1)
    isex = ((ex == t[:, None]) & has).any(1)
    sel = (nex >= 2) & same & (t % 10 == first) & (dist >= 25) & ~isex
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].values[sel].mean())

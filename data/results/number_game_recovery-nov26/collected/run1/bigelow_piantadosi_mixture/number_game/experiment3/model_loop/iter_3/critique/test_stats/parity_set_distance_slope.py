# name: parity_set_distance_slope
# description: Among sets of >=3 examples all of one parity, yes rate for same-parity non-example targets >=25 from the nearest example minus that for same-parity targets <=8 away; observed > null (less negative) means people's parity generalisation is flatter over distance than the model's Gaussian hesitance implies, observed < null that it decays more steeply.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).fillna(0).astype(int).values
    has = ex > 0
    nex = has.sum(1)
    t = df["target"].astype(int).values
    par = np.where(has, ex % 2, -1)
    p0 = par[:, 0]
    same = ((par == p0[:, None]) | ~has).all(1)
    dist = np.where(has, np.abs(ex - t[:, None]), 999).min(1)
    base = (nex >= 3) & same & (t % 2 == p0) & (dist > 0)
    r = df["rating"].values.astype(float)
    far, near = base & (dist >= 25), base & (dist <= 8)
    if far.sum() == 0 or near.sum() == 0:
        return 0.0
    return float(r[far].mean() - r[near].mean())

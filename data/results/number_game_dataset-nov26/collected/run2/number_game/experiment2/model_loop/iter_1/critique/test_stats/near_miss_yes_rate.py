# name: near_miss_yes_rate
# description: Yes-rate on targets 1-3 away from the nearest example (target not itself an example) minus yes-rate on all other non-example targets; observed below null means people reject near neighbours more than the model (which has no proximity term) predicts, above means numeric similarity generalization the model misses.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    near = (dist >= 1) & (dist <= 3)
    other = (~near) & (~inset)
    if near.sum() == 0 or other.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[other].mean())

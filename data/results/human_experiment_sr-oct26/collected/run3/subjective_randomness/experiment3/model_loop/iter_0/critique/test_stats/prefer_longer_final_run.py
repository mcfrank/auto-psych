# name: prefer_longer_final_run
# description: Among same-length pairs whose final runs (identical flips at the end) differ in length, the rate of choosing the sequence with the longer final run; observed below null means people penalise a streak at the end (recency) more than the model, above means less.
def _final_run(s):
    last = s.str[-1]
    n = s.str.len().max()
    run = np.ones(len(s), dtype=int)
    alive = np.ones(len(s), dtype=bool)
    L = s.str.len().to_numpy()
    lastv = last.to_numpy()
    for k in range(2, n + 1):
        ck = s.str[-k].to_numpy()
        ok = alive & (L >= k) & (ck == lastv)
        run = run + ok
        alive = ok
    return run
def test_statistic(df):
    ra = _final_run(df["sequence_a"])
    rb = _final_run(df["sequence_b"])
    m = ra != rb
    y = df["chose_left"].to_numpy()
    chose_longer = np.where(ra > rb, y, 1 - y)
    return float(chose_longer[m].mean())

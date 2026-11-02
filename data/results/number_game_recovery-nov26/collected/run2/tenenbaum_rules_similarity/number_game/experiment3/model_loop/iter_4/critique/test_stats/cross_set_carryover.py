# name: cross_set_carryover
# description: On the first trial after the set changes (excluding session start and trials repeating the previous target), P(yes | previous trial yes) minus P(yes | previous trial no) within a session; observed above null means a response perseveration/momentum across sets that the model (independent trials) under-produces.
def test_statistic(df):
    ns = df["number_set"].astype(str).values
    ti = df["trial_index"].values
    tg = df["target"].values
    r = df["rating"].values.astype(float)
    prev_ns = np.concatenate([[""], ns[:-1]])
    prev_r = np.concatenate([[np.nan], r[:-1]])
    prev_t = np.concatenate([[-1], tg[:-1]])
    m = (ns != prev_ns) & (ti > 0) & (tg != prev_t)
    a = m & (prev_r == 1)
    b = m & (prev_r == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

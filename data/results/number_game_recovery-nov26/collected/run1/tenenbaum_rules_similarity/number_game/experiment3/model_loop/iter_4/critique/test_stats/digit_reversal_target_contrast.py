# name: digit_reversal_target_contrast
# description: Within sets having a two-digit example, yes rate for non-example targets whose digit reversal is an example (e.g. 38 for 83) minus yes rate for other non-example targets of the same sets; observed above null means people use a digit-reversal/same-digits similarity the model lacks, below null means they avoid it more than the model.
def test_statistic(df):
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    rev_flag = np.zeros(len(t), bool); ex_flag = np.zeros(len(t), bool); elig = np.zeros(len(t), bool)
    cache = {}
    for x in set(s):
        v = [int(a) for a in x.split(",")]
        revs = set(int(str(a)[::-1]) for a in v if 10 <= a <= 99 and a % 10 != 0 and a % 11 != 0)
        cache[x] = (set(v), revs)
    ex_sets = [cache[x] for x in s]
    ex_flag = np.array([ti in e[0] for ti, e in zip(t, ex_sets)])
    rev_flag = np.array([ti in e[1] for ti, e in zip(t, ex_sets)])
    elig = np.array([len(e[1]) > 0 for e in ex_sets])
    a = y[elig & rev_flag & ~ex_flag]
    b = y[elig & ~rev_flag & ~ex_flag]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())

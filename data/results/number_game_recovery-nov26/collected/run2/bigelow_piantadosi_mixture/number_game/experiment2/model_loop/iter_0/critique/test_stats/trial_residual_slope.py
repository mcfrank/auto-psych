# name: trial_residual_slope
# description: OLS slope (per 100 trials) of (rating - proxy probability) on trial_index; nonzero observed vs null means yes-responding drifts across the session in a way the stationary model does not capture (positive = people become more liberal than predicted over time).

_CACHE = {}


def _proxy(df):
    """Fixed-parameter stimulus-only proxy of the incumbent: P(target in concept)
    under equal-prior rules + intervals, lambda=0.5, sampling without replacement,
    10% random examples, 3% lapse at 50% yes. Cached because it ignores ratings."""
    key = (len(df), str(df["number_set"].iloc[0]), str(df["number_set"].iloc[-1]), int(df["target"].sum()))
    if key in _CACHE:
        return _CACHE[key]
    nums = np.arange(1, 101)
    rules = []
    rules.append(nums % 2 == 1)
    rules.append(np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in nums]))
    rules.append(np.isin(nums, [k * k for k in range(1, 11)]))
    rules.append(np.isin(nums, [k ** 3 for k in range(1, 5)]))
    rules.append(np.isin(nums, [2, 4, 8, 16, 64]))
    rules.append(np.isin(nums, [2, 4, 8, 16, 32, 37, 64]))
    rules.append(nums % 2 == 0)
    for mm in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        rules.append(nums % mm == 0)
    for mm in range(2, 11):
        rules.append(np.isin(nums, [mm ** k for k in range(1, 8) if mm ** k <= 100]))
    for dd in range(10):
        rules.append(nums % 10 == dd)
    for dd in range(10):
        rules.append(np.array([str(dd) in str(n) for n in nums]))
    lo, hi = np.triu_indices(100)
    ints = (nums[None, :] >= nums[lo][:, None]) & (nums[None, :] <= nums[hi][:, None])
    mem = np.vstack([np.array(rules), ints]).astype(float)
    nr = len(rules)
    size = mem.sum(1)
    logprior = np.concatenate([np.full(nr, np.log(0.5 / nr)), np.full(len(ints), np.log(0.5 / len(ints)))])
    sets = df["number_set"].astype(str).values
    uniq, inv = np.unique(sets, return_inverse=True)
    pin = np.zeros((len(uniq), 100))
    for i, s in enumerate(uniq):
        tot = logprior.copy()
        shown = np.zeros(len(mem))
        for x in [int(v) for v in s.split(",")]:
            m_ = mem[:, x - 1]
            tot += np.log(0.9 * m_ / np.maximum(size - shown, 1.0) + 0.1 / 100)
            shown += m_
        w = np.exp(tot - tot.max())
        pin[i] = (w / w.sum()) @ mem
    q = pin[inv, df["target"].astype(int).values - 1]
    p = 0.97 * q + 0.03 * 0.5
    _CACHE[key] = (q, p)
    return q, p


def _examples(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).values.astype(float)
    return ex


def test_statistic(df):
    q, p = _proxy(df)
    r = df["rating"].values - p
    x = df["trial_index"].astype(float).values / 100.0
    x = x - x.mean()
    return float(np.sum(x * (r - r.mean())) / np.sum(x * x))

# name: single_final_digit_match_residual
# description: Mean of (rating - proxy probability) over non-example targets whose final digit matches exactly one example in a multi-example set; negative observed vs null means the model over-generalizes from a single shared last digit (ends-in / contains-digit rules get too much credit from one matching example).

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
    ex = _examples(df)
    t = df["target"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    same = np.nansum(np.where(np.isnan(ex), 0, (ex % 10) == (t % 10)[:, None]), axis=1)
    inset = np.any(ex == t[:, None], axis=1)
    sel = (same == 1) & (n >= 2) & (~inset)
    if sel.sum() == 0:
        return 0.0
    return float(np.mean(df["rating"].values[sel] - p[sel]))

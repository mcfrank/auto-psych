# name: set_level_residual_chisq
# description: Pearson chi-square of per-set yes counts against incumbent proxy expectations, divided by the number of sets (sum over sets of (obs-exp)^2 / sum p(1-p)); observed above null means sets carry idiosyncratic overall yes-biases the model misses, below null means set-level totals vary less than the model's binomial noise.

_CACHE = {}


def _proxy(df):
    """Stimulus-only proxy of the incumbent at its MAP parameters (rule weight
    0.315, range width prior w**-0.08, no lapse): rules + intervals, examples
    drawn without replacement, 10% stray. Returns P(yes) per row; cached."""
    key = (len(df), str(df["number_set"].iloc[0]), str(df["number_set"].iloc[-1]), int(df["target"].sum()))
    if key in _CACHE:
        return _CACHE[key]
    nums = np.arange(1, 101)
    rules = [nums % 2 == 1,
             np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in nums]),
             np.isin(nums, [k * k for k in range(1, 11)]),
             np.isin(nums, [k ** 3 for k in range(1, 5)]),
             np.isin(nums, [2, 4, 8, 16, 64]),
             np.isin(nums, [2, 4, 8, 16, 32, 37, 64]),
             nums % 2 == 0]
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
    width = size[nr:]
    lw = -0.08 * np.log(width)
    lw = lw - (np.log(np.sum(np.exp(lw - lw.max()))) + lw.max())
    logprior = np.concatenate([np.full(nr, np.log(0.315 / nr)), np.log(0.685) + lw])
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
    p = np.clip(pin[inv, df["target"].astype(int).values - 1], 1e-6, 1 - 1e-6)
    _CACHE[key] = p
    return p


def _features(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).values.astype(float)
    t = df["target"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    lo = np.nanmin(ex, 1)
    hi = np.nanmax(ex, 1)
    inset = (ex == t[:, None]).any(1)
    out = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    return ex, t, n, lo, hi, inset, out


def _res(df, sel):
    p = _proxy(df)
    y = df["rating"].astype(float).values
    if sel.sum() == 0:
        return 0.0
    return float(np.mean(y[sel] - p[sel]))


def test_statistic(df):
    p = _proxy(df)
    g = pd.DataFrame({"s": df["number_set"].astype(str).values, "y": df["rating"].astype(float).values,
                      "p": p, "v": p * (1 - p)})
    c = g.groupby("s").agg(y=("y", "sum"), p=("p", "sum"), v=("v", "sum"))
    return float(np.mean((c["y"] - c["p"]) ** 2 / np.maximum(c["v"], 1e-6)))

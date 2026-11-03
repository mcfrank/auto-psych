# name: rule_contrast_late_minus_early
# description: In computed-rule sets of 2-4 examples, the rule contrast (yes-rate on non-example rule members minus yes-rate on non-members) in the second half of each session minus the first half; observed below the model (fixed search share) means rule search fades with fatigue over the session, above means it grows with practice.
def _set_info(df):
    # Per distinct set: number of examples, span, examples, and the computed rules all examples fit.
    primes = [n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1))]
    rules = [primes, [2, 4, 8, 16, 32, 64], [i * i for i in range(1, 11)], [1, 8, 27, 64]]
    rules += [list(range(m, 101, m)) for m in [3, 4, 6, 7, 8, 9, 11, 12]]
    rules = [set(r) for r in rules]
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    k = len(uniq)
    n = np.zeros(k); lo = np.zeros(k); hi = np.zeros(k)
    member = np.zeros((k, 102), bool); example = np.zeros((k, 102), bool)
    for i, s in enumerate(uniq):
        e = [int(x) for x in s.split(",")]
        n[i], lo[i], hi[i] = len(e), min(e), max(e)
        example[i, e] = True
        for r in rules:
            if all(x in r for x in e):
                member[i, list(r)] = True
    t = df["target"].astype(int).to_numpy()
    computed = member.any(axis=1)[codes]
    return dict(codes=codes, n=n[codes], lo=lo[codes], hi=hi[codes], t=t,
                member=member[codes, t], example=example[codes, t], computed=computed,
                y=df["rating"].astype(float).to_numpy())

def test_statistic(df):
    d = _set_info(df)
    ti = df["trial_index"].to_numpy()
    pid = (df["trial_index"].diff().fillna(0) < 0).cumsum().to_numpy()
    mx = pd.Series(ti).groupby(pid).transform("max").to_numpy()
    late = ti > mx / 2
    base = d["computed"] & (d["n"] >= 2) & ~d["example"]
    y = d["y"]
    out = []
    for h in (late, ~late):
        mem = base & h & d["member"]
        non = base & h & ~d["member"]
        if not mem.any() or not non.any():
            return 0.0
        out.append(y[mem].mean() - y[non].mean())
    return float(out[0] - out[1])

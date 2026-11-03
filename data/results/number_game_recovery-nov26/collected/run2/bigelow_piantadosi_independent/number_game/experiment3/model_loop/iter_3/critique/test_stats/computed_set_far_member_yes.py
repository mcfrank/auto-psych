# name: computed_set_far_member_yes
# description: Yes-rate for targets outside the examples' span that belong to a computed rule shared by all examples (sets of 2-4 examples sharing such a rule); observed below the model means people extend computed rules beyond the span less than predicted (glance/range reasoning dominates more), above means they search for the rule more often than the fixed share.
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
    m = d["computed"] & (d["n"] >= 2) & d["member"] & ~d["example"] & ((d["t"] < d["lo"]) | (d["t"] > d["hi"]))
    return float(d["y"][m].mean()) if m.any() else 0.0

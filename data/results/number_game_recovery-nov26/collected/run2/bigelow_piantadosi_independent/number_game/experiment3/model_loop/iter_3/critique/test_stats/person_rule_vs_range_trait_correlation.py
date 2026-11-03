# name: person_rule_vs_range_trait_correlation
# description: Across participants, the correlation between their cell-demeaned yes residual on out-of-span members of a shared computed rule and on in-span non-members, in computed-rule sets of 2-4 examples; observed more negative than the model (no person effects, ~0) means some people are consistent rule-searchers and others consistent range-followers, i.e. search is a person trait, not a per-judgment coin flip.
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
    pid = (df["trial_index"].diff().fillna(0) < 0).cumsum().to_numpy()
    cell = pd.factorize(df["number_set"].astype(str) + "|" + df["target"].astype(str))[0]
    y = d["y"]
    r = y - pd.Series(y).groupby(cell).transform("mean").to_numpy()
    base = d["computed"] & (d["n"] >= 2) & ~d["example"]
    inspan = (d["t"] >= d["lo"]) & (d["t"] <= d["hi"])
    far = base & d["member"] & ~inspan
    near = base & ~d["member"] & inspan
    a = pd.Series(r[far]).groupby(pid[far]).mean()
    b = pd.Series(r[near]).groupby(pid[near]).mean()
    j = pd.concat([a, b], axis=1, join="inner")
    if len(j) < 3:
        return 0.0
    c = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
    return float(c) if np.isfinite(c) else 0.0

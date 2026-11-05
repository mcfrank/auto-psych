# name: full_match_out_of_span_rate
# description: Yes-rate for targets lying outside the examples' [min,max] span that nonetheless have every parity/prime/square/cube/power/multiple/final-digit/contained-digit property shared by all examples (requires >=1 such shared property); observed > null means the model's span feature penalizes rule-consistent out-of-span targets too much, observed < null too little.
def test_statistic(df):
    N = range(1, 101)
    props = [{n for n in N if n % 2 == 1}, {n for n in N if n % 2 == 0},
             {n for n in N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))},
             {k * k for k in range(1, 11)}, {k ** 3 for k in range(1, 5)}]
    for b in range(2, 11):
        props.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for m in range(3, 13):
        props.append({n for n in N if n % m == 0})
    for d in range(10):
        props.append({n for n in N if n % 10 == d})
    for d in range(10):
        props.append({n for n in N if str(d) in str(n)})
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    keys = pd.Series(sets) + "|" + pd.Series(t.astype(str))
    lut = {}
    for k in pd.unique(keys):
        s, tt = k.split("|")
        xs = [int(x) for x in s.split(",")]
        tt = int(tt)
        if min(xs) <= tt <= max(xs):
            lut[k] = False
            continue
        sh = [p for p in props if all(x in p for x in xs)]
        lut[k] = bool(sh) and all(tt in p for p in sh)
    m = keys.map(lut).values.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())

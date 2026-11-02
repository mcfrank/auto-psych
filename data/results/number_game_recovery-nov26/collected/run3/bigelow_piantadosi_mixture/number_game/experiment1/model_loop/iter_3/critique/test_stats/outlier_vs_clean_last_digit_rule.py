# name: outlier_vs_clean_last_digit_rule
# description: Among 3-4 example sets, yes rate for unseen targets ending in the majority last digit when exactly one example breaks that digit, minus the same rate when all examples share it; observed > null means people discount a rule-breaking example less (keep the rule more) than the model's uncertain-reliability noise produces, observed < null means one stray undermines the rule more than the model predicts.
def test_statistic(df):
    sets = df["number_set"].unique()
    dig = {}
    kind = {}
    for s in sets:
        v = [int(x) for x in str(s).split(",")]
        if len(v) < 3:
            continue
        ld = [x % 10 for x in v]
        vals, cnt = np.unique(ld, return_counts=True)
        k = int(np.argmax(cnt))
        if cnt[k] == len(v):
            kind[s], dig[s] = 0, int(vals[k])
        elif cnt[k] == len(v) - 1:
            kind[s], dig[s] = 1, int(vals[k])
    ns = df["number_set"]
    kd = ns.map(kind)
    dg = ns.map(dig)
    t = df["target"].astype(int)
    # vectorised membership test via string matching on padded set
    padded = "," + ns.astype(str) + ","
    tstr = "," + t.astype(str) + ","
    member = np.array([b.find(a) >= 0 for a, b in zip(tstr.values, padded.values)])
    match = (t % 10 == dg) & ~member
    r = df["rating"].astype(float)
    a = r[match & (kd == 1)]
    b = r[match & (kd == 0)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())

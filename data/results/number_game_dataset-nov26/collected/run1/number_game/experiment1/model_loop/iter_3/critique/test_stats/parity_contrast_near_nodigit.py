# name: parity_contrast_near_nodigit
# description: In single-parity sets (>=2 examples), yes-rate for same-parity minus opposite-parity targets that lie within 10 of some example and share no final digit with any example; observed > null means people reject near opposite-parity targets more sharply (rule-like parity) than the model's constant multiplicative parity penalty predicts, observed < null means parity matters less near the examples than the model assumes.

def _feats(df):
    # per unique (set, target) pair features; loops only over unique pairs
    pairs = df[["number_set", "target"]].drop_duplicates()
    rows = []
    for s, t in zip(pairs["number_set"].values, pairs["target"].values):
        ex = [int(x) for x in str(s).split(",")]
        t = int(t)
        d = [abs(t - e) for e in ex]
        par = set(e % 2 for e in ex)
        digs = [e % 10 for e in ex]
        nearest = ex[int(np.argmin(d))]
        rows.append(dict(number_set=s, target=t, n=len(ex), mind=min(d),
                         dmatch=float(t % 10 in digs),
                         singlepar=float(len(par) == 1),
                         samepar=float(len(par) == 1 and t % 2 in par),
                         nearsamepar=float(nearest % 2 == t % 2),
                         digit_homog=float(len(ex) >= 2 and len(set(digs)) == 1),
                         all3odd=float(all(e % 3 == 0 and e % 2 == 1 for e in ex) and len(ex) >= 2),
                         inrange=float(min(ex) < t < max(ex))))
    f = pd.DataFrame(rows)
    f["target"] = f["target"].astype(int)
    d2 = df[["number_set", "target", "rating"]].copy()
    d2["target"] = d2["target"].astype(int)
    return d2.merge(f, on=["number_set", "target"], how="left")

def _rate(x):
    return float(x["rating"].mean()) if len(x) else 0.0


def test_statistic(df):
    f = _feats(df)
    m = (f["singlepar"] == 1) & (f["n"] >= 2) & (f["mind"] <= 10) & (f["dmatch"] == 0)
    a = f[m & (f["samepar"] == 1)]; b = f[m & (f["samepar"] == 0)]
    return _rate(a) - _rate(b)


# name: majority_parity_exception_contrast
# description: For sets of >=3 examples where exactly one example has the other parity (e.g. 43,4,91,9), mean over sets of (yes rate on non-example targets with the majority parity minus yes rate on targets with the exception's parity); observed > null means people treat the lone odd-one-out as noise and generalise a parity rule more strongly than the model, observed < null means they honour the exception more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    uniq = sets.unique()
    info = {}
    for s in uniq:
        e = [int(x) for x in s.split(",")]
        if len(e) < 3:
            continue
        odd = sum(x % 2 for x in e)
        if odd == 1:
            info[s] = (0, set(e))
        elif odd == len(e) - 1:
            info[s] = (1, set(e))
    out = []
    for s, (maj, es) in info.items():
        m = (sets.values == s)
        tt, rr = t[m], r[m]
        keep = np.array([x not in es for x in tt])
        tt, rr = tt[keep], rr[keep]
        a = rr[(tt % 2) == maj]
        b = rr[(tt % 2) != maj]
        if len(a) and len(b):
            out.append(a.mean() - b.mean())
    return float(np.mean(out)) if out else 0.0

# name: set_size_sharpening
# description: Yes-rate on minimal-rule non-member targets in 4-example sets minus that in 2-example sets; observed above null (less negative) means the model's size principle sharpens generalization with more examples more than humans do.
def _rules():
    N = np.arange(1, 101)
    rs = [N % 2 == 1, N % 2 == 0,
          np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in N]),
          np.isin(N, [k * k for k in range(1, 11)]), np.isin(N, [1, 8, 27, 64]),
          np.isin(N, [2, 4, 8, 16, 64]), np.isin(N, [2, 4, 8, 16, 32, 37, 64])]
    rs += [N % m == 0 for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rs += [np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]) for m in range(2, 11)]
    rs += [N % 10 == d for d in range(10)]
    rs += [np.array([str(d) in str(n) for n in N]) for d in range(10)]
    return np.array(rs, dtype=bool)


def _set_info(df):
    # per row: examples list, minimal consistent rule membership of target, any consistent rule membership
    R = _rules()
    sizes = R.sum(1)
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    min_mem, any_mem, ex = {}, {}, {}
    for s in uniq:
        e = np.array([int(x) for x in s.split(",")])
        ex[s] = e
        cons = R[:, e - 1].all(1)
        if cons.any():
            idx = np.where(cons)[0]
            best = idx[np.argmin(sizes[idx])]
            min_mem[s] = R[best]
            any_mem[s] = R[idx].any(0)
        else:
            min_mem[s] = np.zeros(100, bool)
            any_mem[s] = np.zeros(100, bool)
    t = df["target"].astype(int).to_numpy() - 1
    codes, inv = np.unique(sets.to_numpy(), return_inverse=True)
    MM = np.array([min_mem[c] for c in codes])
    AM = np.array([any_mem[c] for c in codes])
    has_rule = np.array([min_mem[c].any() for c in codes])
    return {"in_min": MM[inv, t], "in_any": AM[inv, t], "has_rule": has_rule[inv],
            "codes": codes, "inv": inv, "ex": [ex[c] for c in codes], "t": t + 1}


def test_statistic(df):
    I = _set_info(df)
    r = df["rating"].to_numpy(float)
    n = df["number_set"].astype(str).str.count(",").to_numpy() + 1
    nm = ~I["in_min"]
    a, b = nm & (n == 4), nm & (n == 2)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())

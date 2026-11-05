# name: inconsistent_target_yes_rate
# description: Yes-rate on targets in no rule consistent with the examples and outside the examples' [min,max] range; observed above null means the model under-produces 'yes' to unrelated targets (humans have a higher guessing/similarity floor).
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
    lo = np.array([e.min() for e in I["ex"]])[I["inv"]]
    hi = np.array([e.max() for e in I["ex"]])[I["inv"]]
    t = I["t"]
    m = (~I["in_any"]) & ((t < lo) | (t > hi))
    return float(r[m].mean()) if m.any() else 0.0

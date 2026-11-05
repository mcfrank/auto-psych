# name: member_gradedness_spread
# description: Standard deviation, across (set,target) cells whose target is in the minimal consistent rule, of the cell mean rating after subtracting the set's member mean; observed above null means humans grade rule members unequally (typicality), while the model treats all members of a concept alike.
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
    d = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy(), "t": I["t"],
                      "r": df["rating"].to_numpy(float), "m": I["in_min"]})
    d = d[d["m"]]
    if len(d) == 0:
        return 0.0
    c = d.groupby(["s", "t"])["r"].mean().reset_index()
    c["dev"] = c["r"] - c.groupby("s")["r"].transform("mean")
    return float(c["dev"].std(ddof=0))

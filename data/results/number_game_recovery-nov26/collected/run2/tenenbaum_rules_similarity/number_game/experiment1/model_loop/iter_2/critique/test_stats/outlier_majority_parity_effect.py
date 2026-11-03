# name: outlier_majority_parity_effect
# description: For sets of 3-4 examples where all but exactly one example share a parity, yes-rate on non-member targets of the majority parity minus those of the minority parity; observed above null means people discount the odd-one-out example and keep the majority-parity rule more than the model's single inferred noise rate allows, below null means they discount it less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    maj = {}
    for s in uniq:
        ex = [int(x) for x in s.split(",")]
        ne = sum(e % 2 == 0 for e in ex)
        if len(ex) >= 3 and ne == len(ex) - 1:
            maj[s] = 0
        elif len(ex) >= 3 and ne == 1:
            maj[s] = 1
        else:
            maj[s] = -1
    p = sets.map(maj).to_numpy()
    mem = {s: set(int(x) for x in s.split(",")) for s in uniq}
    t = df["target"].astype(int).to_numpy()
    # vectorised membership via string key lookup
    keys = sets + "|" + df["target"].astype(int).astype(str)
    memkeys = {f"{s}|{e}" for s in uniq for e in mem[s]}
    nonmem = ~keys.isin(memkeys).to_numpy()
    r = df["rating"].to_numpy()
    ok = (p >= 0) & nonmem
    a = ok & (t % 2 == p)
    b = ok & (t % 2 != p)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

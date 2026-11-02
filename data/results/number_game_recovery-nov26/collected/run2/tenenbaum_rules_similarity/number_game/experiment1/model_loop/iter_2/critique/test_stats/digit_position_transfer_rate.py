# name: digit_position_transfer_rate
# description: For sets of >=2 examples that all end in the same digit d, the yes-rate on targets whose tens digit is d but which do not end in d (e.g. 64,94,84 -> 40-49); observed below null means the model over-generalises via its 'contains digit d' rules (people treat the shared digit as position-specific), observed above null means people transfer the digit across positions more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    dmap = {}
    for s in uniq:
        ex = [int(x) for x in s.split(",")]
        last = {e % 10 for e in ex}
        dmap[s] = (last.pop() if (len(ex) >= 2 and len(last) == 1) else -1)
    d = sets.map(dmap).to_numpy()
    t = df["target"].astype(int).to_numpy()
    mask = (d >= 0) & (t // 10 == d) & (t % 10 != d) & (t < 100)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[mask].mean())

# name: partial_majority_final_digit
# description: In sets of >=3 examples where all but one example share a final digit (not all), yes-rate for targets with that majority final digit minus yes-rate for other targets; observed > null means the model (which only reacts to a digit shared by ALL examples) under-produces endorsement of near-unanimous final-digit patterns.
def test_statistic(df):
    sets = df["number_set"].unique()
    maj = {}
    for s in sets:
        ex = [int(x) for x in str(s).split(",")]
        if len(ex) < 3:
            continue
        ds = [e % 10 for e in ex]
        for d in set(ds):
            if ds.count(d) == len(ex) - 1:
                maj[s] = d
    m = df["number_set"].map(maj)
    sel = m.notna()
    if sel.sum() == 0:
        return 0.0
    sub = df[sel]
    hit = (sub["target"].astype(int) % 10).values == m[sel].astype(int).values
    r = sub["rating"].values.astype(float)
    if hit.sum() == 0 or (~hit).sum() == 0:
        return 0.0
    return float(r[hit].mean() - r[~hit].mean())

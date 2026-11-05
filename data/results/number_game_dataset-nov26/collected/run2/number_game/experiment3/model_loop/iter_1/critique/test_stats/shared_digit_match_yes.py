# name: shared_digit_match_yes
# description: Among multi-example sets whose examples all share a final digit, yes rate on targets that also end in that digit (excluding targets that are themselves examples); observed > null means the model under-produces endorsement of a visible final-digit rule, observed < null over-produces it.
def test_statistic(df):
    s = df["number_set"].astype(str)
    u, inv = np.unique(s.values, return_inverse=True)
    dig = np.full(len(u), -1)
    for i, x in enumerate(u):
        ex = [int(v) for v in x.split(",")]
        ds = {v % 10 for v in ex}
        if len(ex) >= 2 and len(ds) == 1:
            dig[i] = ds.pop()
    d = dig[inv]
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    sel = (d >= 0) & (t % 10 == d)
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())

# name: far_outside_range_yes_rate
# description: Yes-rate for targets more than 20 below the smallest or above the largest example; observed above null means the model generalizes too narrowly beyond the example range (under-produces distant yeses), below means too broadly.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    parsed = {s: np.array([int(x) for x in s.split(",")]) for s in uniq}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = sets.map({s: a.min() for s, a in parsed.items()}).to_numpy()
    hi = sets.map({s: a.max() for s, a in parsed.items()}).to_numpy()
    sel = (t < lo - 20) | (t > hi + 20)
    return float(r[sel].mean()) if sel.any() else 0.0

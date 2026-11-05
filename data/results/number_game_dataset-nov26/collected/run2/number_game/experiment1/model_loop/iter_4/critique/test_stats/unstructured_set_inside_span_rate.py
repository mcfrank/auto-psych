# name: unstructured_set_inside_span_rate
# description: Yes-rate for targets strictly inside [min,max] of sets whose examples do NOT share parity; observed < null means the model over-endorses in-span targets when there is no shared arithmetic rule (its magnitude bounds dominate a near-empty profile), observed > null means people generalize across the span even more than predicted.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    lo, hi, mixed = {}, {}, {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        lo[s], hi[s] = min(ex), max(ex)
        mixed[s] = len({e % 2 for e in ex}) > 1
    t = df["target"].astype(int).to_numpy()
    sel = sets.map(mixed).to_numpy().astype(bool) & (t > sets.map(lo).to_numpy()) & (t < sets.map(hi).to_numpy())
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[sel].mean())

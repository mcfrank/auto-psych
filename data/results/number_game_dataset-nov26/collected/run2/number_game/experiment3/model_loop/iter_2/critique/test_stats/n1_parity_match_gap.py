# name: n1_parity_match_gap
# description: For one-example sets, yes rate on same-parity targets minus opposite-parity targets; observed > null means the model's flat-rate doubt still dilutes parity generalization from a lone example (observed < null: model over-weights parity).
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    e = pd.to_numeric(s.where(one, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    same = one & (t % 2 == e % 2)
    diff = one & (t % 2 != e % 2)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())

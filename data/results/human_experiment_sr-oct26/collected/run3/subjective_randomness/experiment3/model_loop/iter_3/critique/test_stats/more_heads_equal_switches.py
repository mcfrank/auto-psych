# name: more_heads_equal_switches
# description: Among pairs with equal switch counts but different heads counts, the rate of choosing the sequence with more heads; observed below null means people avoid head-heavy sequences more than the model's heads-rigged trick-coin prior predicts (when switching cannot explain it), above means the model over-penalises heads.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    ha = a.str.count("H"); hb = b.str.count("H")
    m = ((ka == kb) & (ha != hb)).to_numpy()
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    left_more = (ha > hb).to_numpy()[m]
    return float(np.mean(np.where(left_more, y, 1 - y)))

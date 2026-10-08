# name: more_balanced_choice_equal_switches
# description: Among pairs with equal switch counts but different heads imbalance |H - n/2|, the rate of choosing the more balanced sequence; observed above null means people penalise lopsided H/T composition more than the model's trick-coin explanation does, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    n = a.str.len()
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    ia = (a.str.count("H") - n / 2.0).abs(); ib = (b.str.count("H") - n / 2.0).abs()
    m = ((ka == kb) & (ia != ib)).to_numpy()
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    left_bal = (ia < ib).to_numpy()[m]
    return float(np.mean(np.where(left_bal, y, 1 - y)))

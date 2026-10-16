# name: equal_switch_item_rate_variance
# description: Across length-8 pairs with equal switch counts, the variance of the item-level rate of choosing the alphabetically-first sequence (side-independent); observed above null means people discriminate among equal-switch sequences more sharply than the model's motif/second-order/trick-coin structure predicts, below means the model over-discriminates them.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    ka = a.str.count("HT") + a.str.count("TH"); kb = b.str.count("HT") + b.str.count("TH")
    n = a.str.len()
    m = ((n == 8) & (ka == kb)).to_numpy()
    if m.sum() == 0:
        return 0.0
    av = a.to_numpy()[m]; bv = b.to_numpy()[m]
    first_is_a = av < bv
    y = df["chose_left"].to_numpy()[m]
    c = np.where(first_is_a, y, 1 - y)
    key = np.where(first_is_a, av + "|" + bv, bv + "|" + av)
    r = pd.Series(c).groupby(key).mean()
    return float(r.var(ddof=0)) if len(r) > 1 else 0.0

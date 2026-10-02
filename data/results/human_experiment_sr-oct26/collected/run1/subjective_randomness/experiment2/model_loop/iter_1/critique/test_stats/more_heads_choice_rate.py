# name: more_heads_choice_rate
# description: Among trials whose sequences differ in their number of heads, the proportion choosing the sequence with MORE heads; the model's imbalance term is symmetric in H/T, so observed away from the null mean signals an asymmetric heads (above) or tails (below) preference it cannot express.
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy()
    hb = df["sequence_b"].str.count("H").to_numpy()
    sel = ha != hb
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(ha > hb, c, 1 - c)
    return float(chose[sel].mean())

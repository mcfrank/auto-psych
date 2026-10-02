# name: more_heads_choice_rate
# description: Among trials where the sequences differ in number of H, the fraction choosing the sequence with more heads; the model is H/T symmetric, so observed away from null means a heads-vs-tails asymmetry (e.g. >null = heads-heavy sequences look more random) that it cannot produce.
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy()
    hb = df["sequence_b"].str.count("H").to_numpy()
    y = df["chose_left"].to_numpy()
    mask = ha != hb
    if mask.sum() == 0:
        return 0.5
    chose_more_h = np.where(ha > hb, y, 1 - y)
    return float(chose_more_h[mask].mean())

# name: more_heads_choice
# description: Among pairs whose head counts differ, the proportion choosing the sequence with MORE heads; the model is symmetric under swapping H and T, so observed far from null_mean means a heads/tails asymmetry (above: heads-heavy sequences look more random than predicted).
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy(); hb = df["sequence_b"].str.count("H").to_numpy()
    sel = ha != hb
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    return float(np.where(ha > hb, c, 1 - c)[sel].mean())

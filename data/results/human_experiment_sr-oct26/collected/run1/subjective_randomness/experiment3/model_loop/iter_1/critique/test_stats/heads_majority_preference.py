# name: heads_majority_preference
# description: Among trials whose two sequences differ in their count of H, the proportion choosing the sequence with MORE heads; observed above null_mean means people find heads-heavy sequences more random than the H/T-symmetric model predicts (below: tails-heavy favoured).
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy()
    hb = df["sequence_b"].str.count("H").to_numpy()
    y = df["chose_left"].to_numpy()
    m = ha != hb
    if m.sum() == 0:
        return 0.5
    chose_more_h = np.where(ha[m] > hb[m], y[m], 1 - y[m])
    return float(chose_more_h.mean())

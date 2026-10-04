# name: more_heads_chosen_rate
# description: Among pairs whose sequences differ in number of H, the rate of choosing the sequence with more heads; the model is H/T-symmetric (predicts ~0.5), so observed above null means people favour heads-heavy sequences (model under-produces an H preference), below means tails-heavy.
def test_statistic(df):
    ha = df["sequence_a"].str.count("H").to_numpy()
    hb = df["sequence_b"].str.count("H").to_numpy()
    m = ha != hb
    c = df["chose_left"].to_numpy()[m]
    picked_more_h = np.where(ha[m] > hb[m], c, 1 - c)
    return float(picked_more_h.mean()) if m.any() else 0.5

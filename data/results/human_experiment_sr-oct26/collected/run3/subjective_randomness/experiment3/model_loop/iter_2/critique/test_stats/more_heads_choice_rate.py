# name: more_heads_choice_rate
# description: Among pairs whose sequences differ in heads count, the rate of choosing the sequence with more heads; observed below null means people avoid head-heavy sequences more than the rigged-trick-coin prior predicts, above means the model over-penalises head-heavy sequences.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ha = a.str.count("H").values; hb = b.str.count("H").values
    y = df["chose_left"].values
    m = ha != hb
    if not m.any():
        return 0.5
    chose_more = np.where(ha > hb, y, 1 - y)[m]
    return float(chose_more.mean())

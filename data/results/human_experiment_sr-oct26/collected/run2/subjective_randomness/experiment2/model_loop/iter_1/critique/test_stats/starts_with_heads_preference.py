# name: starts_with_heads_preference
# description: Among pairs whose first flips differ, the rate of choosing the sequence that starts with H; observed above null means people favour H-first sequences more than the model's overall heads-share term implies (below null: the reverse).
def test_statistic(df):
    fa = df["sequence_a"].str[0].to_numpy(); fb = df["sequence_b"].str[0].to_numpy()
    m = fa != fb
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.where(fa[m] == "H", c, 1 - c).mean())

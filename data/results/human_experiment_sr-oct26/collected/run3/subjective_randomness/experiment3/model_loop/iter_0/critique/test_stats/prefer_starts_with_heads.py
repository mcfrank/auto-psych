# name: prefer_starts_with_heads
# description: Among pairs where exactly one sequence starts with H, the rate of choosing the H-starting sequence as more random; observed above null means people favour sequences opening with Heads more than the model predicts (a first-flip face effect the model lacks), below means they favour T-openings.
def test_statistic(df):
    a0 = df["sequence_a"].str[0].to_numpy()
    b0 = df["sequence_b"].str[0].to_numpy()
    m = (a0 == "H") != (b0 == "H")
    y = df["chose_left"].to_numpy()
    chose_h = np.where(a0 == "H", y, 1 - y)
    if m.sum() == 0:
        return 0.5
    return float(chose_h[m].mean())

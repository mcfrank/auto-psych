# name: pair_choice_variance
# description: Variance across distinct stimulus pairs of the proportion of participants choosing the pair's alphabetically-first sequence; observed above the null means human item-level preferences are more extreme/consistent than the model reproduces (it under-predicts how decisive some pairs are), below means the model over-predicts decisiveness.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b)
    second = np.where(a < b, b, a)
    c = df["chose_left"].to_numpy()
    chose_first = np.where(a < b, c, 1 - c)
    g = pd.DataFrame({"k": pd.Series(first) + "|" + pd.Series(second), "y": chose_first})
    p = g.groupby("k")["y"].mean()
    return float(p.var(ddof=0))

# name: switch_preference_participant_variance
# description: Variance across participants of each person's rate of choosing the sequence with more switches (pairs with unequal switch counts); observed above null means individuals differ in switching preference more than the model's person-level weights allow (below null: less).
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    sa = df["sequence_a"].map(sw).to_numpy(); sb = df["sequence_b"].map(sw).to_numpy()
    m = sa != sb
    y = np.where(sa > sb, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    g = pd.DataFrame({"p": df["participant_id"].to_numpy()[m], "y": y[m]})
    return float(g.groupby("p")["y"].mean().var())

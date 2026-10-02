# name: side_bias_vs_agreement_correlation
# description: Across participants, correlation between strength of side bias |P(Left)-0.5| and majority-agreement rate; observed more negative than null_mean means disengaged participants press one button rather than guessing evenly (the model's lapses are side-neutral coin flips).
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    chose_first = np.where(sw, cl, 1 - cl)
    pair_rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    agree = (chose_first == (pair_rate >= 0.5)).astype(float)
    pid = df["participant_id"].to_numpy()
    g = pd.DataFrame({"agree": agree, "left": cl}).groupby(pid).mean()
    x = (g["left"] - 0.5).abs().to_numpy(); y = g["agree"].to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

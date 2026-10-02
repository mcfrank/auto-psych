# name: agreement_side_bias_interaction
# description: Across participants, Pearson correlation between majority-agreement rate and absolute side bias |Left rate - 0.5|; observed below null_mean means people who agree less with the consensus do so by pressing one button (a side-lapse mechanism the model's symmetric guessing lacks), above means side-biased people are also the consensus followers.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    cf = np.where(sw, cl, 1 - cl)
    pr = pd.Series(cf).groupby(key).transform("mean").to_numpy()
    agree = (cf == (pr >= 0.5)).astype(float)
    t = pd.DataFrame({"p": df["participant_id"].to_numpy(), "g": agree, "c": cl}).groupby("p").mean()
    x = t["g"].to_numpy(); y = np.abs(t["c"].to_numpy() - 0.5)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

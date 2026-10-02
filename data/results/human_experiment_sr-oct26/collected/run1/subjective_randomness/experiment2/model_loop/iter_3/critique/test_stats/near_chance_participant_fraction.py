# name: near_chance_participant_fraction
# description: Fraction of participants whose agreement with the pair-wise majority choice is below 0.6 (near-chance responders; majority from the same dataset, ties skipped); observed above the null means the data hold more disengaged/idiosyncratic people than the model's lapse and sensitivity distributions produce, below means the model over-produces them.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    first = np.where(a < b, a, b)
    key = pd.Series(np.char.add(np.char.add(first, "|"), np.where(a < b, b, a)))
    chose_first = (np.where(df["chose_left"].to_numpy() == 1, a, b) == first).astype(float)
    rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    ok = rate != 0.5
    agree = np.where(rate > 0.5, chose_first, 1 - chose_first)
    s = pd.Series(agree[ok]).groupby(df["participant_id"].to_numpy()[ok]).mean()
    return float((s < 0.6).mean())

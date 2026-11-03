# name: new_set_first_trial_effect
# description: Yes-rate on trials whose number_set differs from the set on the preceding trial_index (first query after a set change) minus yes-rate on trials repeating the previous trial's set; the model is history-free, so observed below null means people are more conservative right after a new set appears (and loosen with repeated queries), above null the reverse.
def test_statistic(df):
    sets = df.groupby("trial_index")["number_set"].first().sort_index()
    prev = sets.shift(1)
    is_new = (sets != prev)
    flag = df["trial_index"].map(is_new).to_numpy(dtype=bool)
    r = df["rating"].astype(float).to_numpy()
    if flag.sum() == 0 or (~flag).sum() == 0:
        return 0.0
    return float(r[flag].mean() - r[~flag].mean())

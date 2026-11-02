# name: lag1_response_repetition
# description: Proportion of consecutive trials (same participant, trial_index increasing by 1) with the same response as the previous trial; observed > null means people perseverate on their previous answer beyond what stimulus-driven model predictions produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].astype(int).to_numpy()
    cons = np.diff(ti) == 1
    if cons.sum() == 0:
        return 0.0
    return float((r[1:] == r[:-1])[cons].mean())

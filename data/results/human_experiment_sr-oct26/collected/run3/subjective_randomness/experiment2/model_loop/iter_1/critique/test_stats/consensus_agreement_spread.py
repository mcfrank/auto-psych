# name: consensus_agreement_spread
# description: SD across participants of each person's rate of agreeing with the item-level majority choice (pairs keyed order-free); observed > null means the model under-produces individual differences in consistency/noise (e.g. some people near-random, others highly consensual).
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy().astype(bool)
    lo = np.where(a < b, a, b); hi = np.where(a < b, b, a)
    chosen = np.where(y, a, b)
    key = pd.Series(lo + "|" + hi)
    chose_lo = pd.Series((chosen == lo).astype(float))
    item_rate = chose_lo.groupby(key).transform("mean").to_numpy()
    maj_lo = item_rate >= 0.5
    agree = (chose_lo.to_numpy() == maj_lo).astype(float)
    return float(pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean().std())

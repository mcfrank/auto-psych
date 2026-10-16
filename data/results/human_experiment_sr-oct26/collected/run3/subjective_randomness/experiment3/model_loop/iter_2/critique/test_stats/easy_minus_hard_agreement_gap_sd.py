# name: easy_minus_hard_agreement_gap_sd
# description: SD across participants of (agreement with majority on clear-cut pairs, majority share >= 0.75) minus (agreement on contested pairs, majority share < 0.65); observed above null means people differ in how their consistency depends on pair difficulty more than a single per-person signed sensitivity scaling all evidence allows (e.g. lapses that cap clear-cut agreement in some people), below means the model over-produces that variation.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    pick_first = np.where(a <= b, df["chose_left"].values, 1 - df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    rate = pd.Series(pick_first).groupby(key.values).transform("mean").values
    maj = np.maximum(rate, 1 - rate)
    agree = np.where(rate >= 0.5, pick_first, 1 - pick_first).astype(float)
    pid = df["participant_id"].values
    easy = pd.Series(np.where(maj >= 0.75, agree, np.nan)).groupby(pid).mean()
    hard = pd.Series(np.where(maj < 0.65, agree, np.nan)).groupby(pid).mean()
    gap = (easy - hard).dropna()
    if len(gap) < 2:
        return 0.0
    return float(gap.std())

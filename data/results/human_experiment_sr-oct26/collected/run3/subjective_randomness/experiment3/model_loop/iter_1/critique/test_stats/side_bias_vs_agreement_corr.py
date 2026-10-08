# name: side_bias_vs_agreement_corr
# description: Across participants, correlation between side extremity |Left rate - 0.5| and majority-agreement rate; observed more negative than null means low-consistency people are side-pressers (disengagement shows as button habit, not coin-flip lapses as the model assumes), more positive means the reverse.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    pick_first = np.where(a <= b, df["chose_left"].values, 1 - df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    rate = pd.Series(pick_first).groupby(key.values).transform("mean").values
    agree = np.where(rate >= 0.5, pick_first, 1 - pick_first)
    pid = df["participant_id"].values
    ag = pd.Series(agree).groupby(pid).mean()
    side = (pd.Series(df["chose_left"].values).groupby(pid).mean() - 0.5).abs()
    if ag.std() == 0 or side.std() == 0:
        return 0.0
    return float(np.corrcoef(ag.values, side.values)[0, 1])

# name: participant_alternation_pref_sd
# description: Across-participant SD of each person's rate of choosing the more-alternating sequence (trials whose alternation counts differ); observed > null means the single-population model under-produces individual differences in alternation preference.
def test_statistic(df):
    def alts(s):
        return s.map({q: sum(q[i] != q[i - 1] for i in range(1, len(q))) for q in s.unique()})
    aa = alts(df["sequence_a"]); ab = alts(df["sequence_b"])
    m = (aa != ab).to_numpy()
    chose_more = np.where(aa.to_numpy() > ab.to_numpy(), df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    r = pd.Series(chose_more[m]).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(r.std(ddof=1)) if len(r) > 1 else 0.0

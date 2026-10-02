# name: alternation_preference_participant_sd
# description: SD across participants of each person's rate of choosing the more alternating sequence (trials where alternation rates differ); observed above the null means people differ in the direction/strength of their switching preference more than the model's personal ideal rates and sensitivities produce, below means the model over-produces that heterogeneity.
def test_statistic(df):
    def alt(s):
        return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: alt(s) for s in u}
    aa = df["sequence_a"].map(m).to_numpy(); ab = df["sequence_b"].map(m).to_numpy()
    sel = np.abs(aa - ab) > 1e-9
    chose_more = np.where(aa > ab, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    s = pd.Series(chose_more[sel]).groupby(df["participant_id"].to_numpy()[sel]).mean()
    return float(s.std())

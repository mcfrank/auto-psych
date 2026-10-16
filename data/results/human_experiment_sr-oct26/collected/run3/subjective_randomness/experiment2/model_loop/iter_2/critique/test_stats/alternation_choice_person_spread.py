# name: alternation_choice_person_spread
# description: SD across participants (with >= 4 such trials) of the rate of choosing the perfectly alternating sequence (length >= 4, HTHT...) when exactly one of the pair is perfectly alternating; observed > null means the model under-produces individual differences in treating perfect alternation as random vs as a pattern, < null that it over-produces them.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def is_alt(s):
        n = s.str.len()
        return (n >= 4) & (s.str.contains("HH|TT", regex=True) == False)
    aa = is_alt(a).to_numpy(); ab = is_alt(b).to_numpy()
    m = aa ^ ab
    y = df["chose_left"].to_numpy()
    chose_alt = np.where(aa, y, 1 - y)[m].astype(float)
    pid = df["participant_id"].to_numpy()[m]
    g = pd.Series(chose_alt).groupby(pid)
    r = g.mean()[g.size() >= 4]
    return float(r.std())

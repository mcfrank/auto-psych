# name: switch_preference_person_spread
# description: SD across participants of each person's rate of choosing the sequence with more switches, among trials with unequal switch counts where neither sequence is perfectly alternating; observed > null means the model under-produces individual differences in switch (alternation vs streak) preference, < null that it over-produces them.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def switches(s):
        n = s.str.len().max()
        p = s.str.pad(n, side="right", fillchar=".")
        tot = np.zeros(len(s))
        for i in range(n - 1):
            c1 = p.str[i].to_numpy(); c2 = p.str[i + 1].to_numpy()
            tot += ((c1 != c2) & (c1 != ".") & (c2 != ".")).astype(float)
        return tot
    ka = switches(a); kb = switches(b)
    la = a.str.len().to_numpy().astype(float)
    alt_a = ka == la - 1; alt_b = kb == la - 1
    m = (ka != kb) & ~alt_a & ~alt_b
    y = df["chose_left"].to_numpy()
    chose_more = np.where(ka > kb, y, 1 - y)[m].astype(float)
    pid = df["participant_id"].to_numpy()[m]
    return float(pd.Series(chose_more).groupby(pid).mean().std())

# name: perfect_alternation_choice_rate
# description: Proportion of choices for a perfectly alternating sequence (HTHT.../THTH..., length>=4) when paired with a non-alternating one; observed above null_mean means the model over-penalises perfect alternation (people find it more random than predicted), below means it under-penalises it.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    def alt(s):
        n = s.str.len()
        p1 = s.str.match(r"^(HT)*H?$"); p2 = s.str.match(r"^(TH)*T?$")
        return (p1 | p2) & (n >= 4)
    aa, ab = alt(a), alt(b)
    m = aa ^ ab
    if m.sum() == 0:
        return 0.5
    chose_alt = np.where(aa[m], df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(chose_alt))

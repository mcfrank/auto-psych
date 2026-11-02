# name: lag1_rating_agreement
# description: Proportion of consecutive trials (trial_index increasing by 1, same set) with the same yes/no answer; observed above the null means sequential response perseveration (or per-set stickiness) the trial-independent model lacks, below means alternation.
def test_statistic(df):
    ti = df["trial_index"].values
    r = df["rating"].values
    s = df["number_set"].astype(str).values
    m = (ti[1:] == ti[:-1] + 1) & (s[1:] == s[:-1])
    if m.sum() == 0:
        return 0.0
    return float((r[1:][m] == r[:-1][m]).mean())

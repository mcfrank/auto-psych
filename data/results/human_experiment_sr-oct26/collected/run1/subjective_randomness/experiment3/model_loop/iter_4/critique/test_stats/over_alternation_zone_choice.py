# name: over_alternation_zone_choice
# description: Among pairs where both sequences have a switch rate of at least 0.6 and the rates differ, the proportion of choices of the MORE-switching sequence; observed above null_mean means people tolerate/prefer over-alternation more than the model's asymmetric ideal-rate penalty predicts, below means they penalise over-alternation more strongly.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    n = a.str.len().to_numpy() - 1.0
    ra = (a.str.count("HT").to_numpy() + a.str.count("TH").to_numpy()) / n
    rb = (b.str.count("HT").to_numpy() + b.str.count("TH").to_numpy()) / n
    m = (ra >= 0.6) & (rb >= 0.6) & (ra != rb)
    if m.sum() == 0:
        return 0.5
    cl = df["chose_left"].to_numpy()
    more = np.where(ra > rb, cl == 1, cl == 0)
    return float(more[m].mean())

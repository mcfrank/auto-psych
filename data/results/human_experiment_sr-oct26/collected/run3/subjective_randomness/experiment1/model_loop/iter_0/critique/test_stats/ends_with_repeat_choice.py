# name: ends_with_repeat_choice
# description: Among pairs where exactly one sequence ends with a repeat (last two flips equal), the rate of choosing that sequence; observed < null means people penalise a terminal repetition (recency/gambler's-fallacy salience) more than the model's position-blind features predict.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ra = (a.str[-1] == a.str[-2]).values; rb = (b.str[-1] == b.str[-2]).values
    m = ra != rb
    if m.sum() == 0:
        return 0.5
    ch = np.where(ra, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(ch[m].mean())

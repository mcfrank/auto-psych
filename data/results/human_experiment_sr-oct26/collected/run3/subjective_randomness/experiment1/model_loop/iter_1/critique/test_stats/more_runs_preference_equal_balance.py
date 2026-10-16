# name: more_runs_preference_equal_balance
# description: Among pairs whose two sequences have the same global H count but different numbers of runs, the proportion of choices for the sequence with MORE runs (more alternations); observed above null_mean means people prefer alternation more than the model's peaked theta_alt irregularity term predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ha = a.str.count("H").values; hb = b.str.count("H").values
    seqs = pd.unique(np.concatenate([a.values, b.values]))
    runs = {s: 1 + sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) for s in seqs}
    ra = a.map(runs).values; rb = b.map(runs).values
    mask = (ha == hb) & (ra != rb)
    if mask.sum() == 0:
        return 0.5
    y = df["chose_left"].values[mask]
    return float(np.where(ra[mask] > rb[mask], y, 1 - y).mean())

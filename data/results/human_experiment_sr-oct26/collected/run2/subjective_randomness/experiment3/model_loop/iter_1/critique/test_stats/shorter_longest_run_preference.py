# name: shorter_longest_run_preference
# description: Among pairs whose longest same-face run differs, the rate of choosing the sequence with the shorter longest run; observed above null means the model under-penalises long streaks, below means it over-penalises them.
def test_statistic(df):
    import re
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    lr = {s: max(len(r) for r in re.findall(r"H+|T+", s)) for s in seqs}
    ra = df["sequence_a"].map(lr).to_numpy()
    rb = df["sequence_b"].map(lr).to_numpy()
    m = ra != rb
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(ra[m] < rb[m], c, 1 - c)))

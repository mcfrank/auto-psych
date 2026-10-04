# name: shorter_longest_run_equal_switches
# description: Among pairs with equal switch counts but different longest runs, the rate of choosing the sequence with the shorter longest run; observed above null means people penalise long streaks beyond what switch rate/span/chunks capture (model under-produces streak aversion), below means the model overstates it.
import re
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    lr = {s: max(len(r) for r in re.findall(r"H+|T+", s)) for s in seqs}
    sw = {s: sum(1 for x, y in zip(s, s[1:]) if x != y) for s in seqs}
    la = df["sequence_a"].map(lr).to_numpy(); lb = df["sequence_b"].map(lr).to_numpy()
    sa = df["sequence_a"].map(sw).to_numpy(); sb = df["sequence_b"].map(sw).to_numpy()
    m = (sa == sb) & (la != lb)
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.where(la[m] < lb[m], c, 1 - c).mean())

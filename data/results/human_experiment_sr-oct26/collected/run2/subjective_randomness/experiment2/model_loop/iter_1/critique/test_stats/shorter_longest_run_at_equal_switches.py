# name: shorter_longest_run_at_equal_switches
# description: Among pairs with equal switch counts but different longest-run lengths, the rate of choosing the sequence with the shorter longest run; observed above null means people penalise long streaks beyond what switching rate predicts (model under-produces streak avoidance).
import re
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    u = pd.unique(np.concatenate([a, b]))
    lr = {s: max(len(m) for m in re.findall("H+|T+", s)) for s in u}
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in u}
    lra = df["sequence_a"].map(lr).to_numpy(); lrb = df["sequence_b"].map(lr).to_numpy()
    swa = df["sequence_a"].map(sw).to_numpy(); swb = df["sequence_b"].map(sw).to_numpy()
    m = (swa == swb) & (lra != lrb)
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    chose_shorter = np.where(lra[m] < lrb[m], c, 1 - c)
    return float(chose_shorter.mean())

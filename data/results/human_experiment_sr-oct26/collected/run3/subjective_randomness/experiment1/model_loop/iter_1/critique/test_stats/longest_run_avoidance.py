# name: longest_run_avoidance
# description: Among pairs whose longest runs differ by >=2 flips, the proportion of choices for the sequence with the SHORTER longest run; observed above null_mean means the model under-penalises long streaks (it has no run-length term beyond windows of 4), below means it over-penalises them.
import re
def test_statistic(df):
    seqs = pd.unique(np.concatenate([df["sequence_a"].values, df["sequence_b"].values]))
    mr = {s: max(len(m) for m in re.findall(r"H+|T+", s)) for s in seqs}
    ra = df["sequence_a"].map(mr).values
    rb = df["sequence_b"].map(mr).values
    d = ra - rb
    mask = np.abs(d) >= 2
    if mask.sum() == 0:
        return 0.5
    chose_shorter = np.where(d[mask] < 0, df["chose_left"].values[mask], 1 - df["chose_left"].values[mask])
    return float(chose_shorter.mean())

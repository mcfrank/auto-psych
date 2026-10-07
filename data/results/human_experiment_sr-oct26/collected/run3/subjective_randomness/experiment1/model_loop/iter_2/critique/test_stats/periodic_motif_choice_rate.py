# name: periodic_motif_choice_rate
# description: Among pairs (length>=6) where exactly one sequence is periodic with period 3 or 4 (e.g. HTHHTHHT, HHTTHHTT) and not perfectly alternating, the proportion of choices for the periodic sequence; observed below null_mean means people penalise repeating motifs more than the model (which sees only switch count and balance) predicts, above means less.
def test_statistic(df):
    def periodic(s):
        n = len(s)
        if n < 6 or all(s[i] != s[i + 1] for i in range(n - 1)):
            return False
        return any(all(s[i] == s[i + p] for i in range(n - p)) for p in (3, 4))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: periodic(s) for s in u}
    pa = df["sequence_a"].map(m).astype(bool)
    pb = df["sequence_b"].map(m).astype(bool)
    sel = pa ^ pb
    if sel.sum() == 0:
        return 0.5
    chose_periodic = np.where(pa[sel], df["chose_left"][sel], 1 - df["chose_left"][sel])
    return float(np.mean(chose_periodic))

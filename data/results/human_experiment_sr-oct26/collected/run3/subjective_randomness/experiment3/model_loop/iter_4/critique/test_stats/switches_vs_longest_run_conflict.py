# name: switches_vs_longest_run_conflict
# description: Among length-8 pairs where the sequence with more switches also contains the strictly longer longest run (switch count and streak length point in opposite directions), the rate of choosing the more-switching sequence; observed above null means the model over-penalises a single long streak relative to overall switching (gambler's run term too strong), below means people weigh the long streak more than the model does.
def test_statistic(df):
    import re
    a = df["sequence_a"]; b = df["sequence_b"]
    uniq = pd.unique(pd.concat([a, b]))
    lr = {s: max(len(r) for r in re.findall(r"H+|T+", s)) for s in uniq}
    la = a.map(lr).to_numpy(); lb = b.map(lr).to_numpy()
    ka = (a.str.count("HT") + a.str.count("TH")).to_numpy()
    kb = (b.str.count("HT") + b.str.count("TH")).to_numpy()
    n = a.str.len().to_numpy()
    m = (n == 8) & (((ka > kb) & (la > lb)) | ((kb > ka) & (lb > la)))
    y = df["chose_left"].to_numpy()[m]
    ta = (ka > kb)[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))

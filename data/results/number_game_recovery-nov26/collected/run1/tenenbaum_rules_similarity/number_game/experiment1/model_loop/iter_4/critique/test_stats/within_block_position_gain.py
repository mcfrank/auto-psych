# name: within_block_position_gain
# description: Within runs of consecutive trials showing the same set (run length >= 4), yes-rate on the 4th-and-later targets minus yes-rate on the first target of the run; observed > null means people grow more liberal as they keep answering about the same set (warm-up/anchoring), which the trial-independent model does not produce beyond stimulus differences.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    ti = df["trial_index"].astype(int)
    new = (ns != ns.shift()) | (ti != ti.shift() + 1)
    blk = new.cumsum()
    pos = df.groupby(blk).cumcount()
    ln = blk.map(blk.value_counts())
    r = df["rating"].astype(float)
    a = (pos == 0) & (ln >= 4)
    b = (pos >= 3) & (ln >= 4)
    if not a.any() or not b.any():
        return 0.0
    return float(r[b].mean() - r[a].mean())

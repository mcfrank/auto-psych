# name: parity_span_interaction
# description: For same-parity sets of >=2 examples, interaction contrast (in-span minus out-of-span yes rate for parity-congruent targets) minus (same for incongruent targets), span = [min, max] of the set; observed above null means people combine parity and range into one concept (parity-within-range) more than the model's additive rule/range mixture, below null means the range boost is concentrated on off-parity targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = (~np.isnan(ex)).sum(axis=1)
    odd = np.nansum(ex % 2 == 1, axis=1)
    allodd = odd == n
    alleven = odd == 0
    t = df["target"].astype(int).to_numpy()
    inr = (t >= np.nanmin(ex, axis=1)) & (t <= np.nanmax(ex, axis=1))
    cong = np.where(allodd, t % 2 == 1, t % 2 == 0)
    sel = (n >= 2) & (allodd | alleven)
    r = df["rating"].to_numpy().astype(float)
    def m(mask):
        k = sel & mask
        return r[k].mean() if k.any() else 0.0
    return float((m(inr & cong) - m(~inr & cong)) - (m(inr & ~cong) - m(~inr & ~cong)))

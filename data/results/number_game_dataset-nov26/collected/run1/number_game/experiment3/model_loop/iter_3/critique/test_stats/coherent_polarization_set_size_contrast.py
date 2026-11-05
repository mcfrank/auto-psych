# name: coherent_polarization_set_size_contrast
# description: Among cells from coherent sets (all same parity, all multiples of 5, or all same final digit), fraction of polarized cells (yes-rate <=1/9 or >=8/9) in 4-example sets minus that in 2-example sets; observed < null means the model's coincidence-driven decisiveness grows too fast with the number of examples (people are not more all-or-none after 4 coherent examples than after 2), observed > null means it grows too slowly.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    par = s % 2
    same_par = par.max(axis=1) == par.min(axis=1)
    dig = s % 10
    same_dig = dig.max(axis=1) == dig.min(axis=1)
    f5 = ((s % 5 == 0) | s.isna()).all(axis=1)
    coh = (n >= 2) & (same_par | same_dig | f5)
    d = df.assign(_n=n.values)[coh.values]
    g = d.groupby(["number_set", "target"]).agg(y=("rating", "mean"), k=("_n", "first"))
    pol = (g["y"] <= 1/9 + 1e-9) | (g["y"] >= 8/9 - 1e-9)
    a = pol[g["k"] == 4]
    b = pol[g["k"] == 2]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())

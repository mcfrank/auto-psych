# name: incoherent_multi_cell_polarization
# description: Among (set, target) cells from 2-4-example sets with no coincidence (examples differ in parity, are not all multiples of 5, and do not share a final digit), fraction of cells with yes-rate <=1/9 or >=8/9; observed > null means people are all-or-none even after incoherent multi-example sets (decisiveness not confined to coherent sets), observed < null means the model over-polarizes them.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    par = s % 2
    same_par = par.max(axis=1) == par.min(axis=1)
    dig = s % 10
    same_dig = dig.max(axis=1) == dig.min(axis=1)
    f5 = ((s % 5 == 0) | s.isna()).all(axis=1)
    inc = (n >= 2) & ~same_par & ~same_dig & ~f5
    d = df[inc.values]
    y = d.groupby(["number_set", "target"])["rating"].mean()
    if len(y) == 0:
        return 0.0
    return float(((y <= 1/9 + 1e-9) | (y >= 8/9 - 1e-9)).mean())

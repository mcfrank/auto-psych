# name: parity_mismatch_yes_rate
# description: Among sets of 2+ examples that all share one parity, the yes-rate for targets of the opposite parity; observed > model means people use parity less strictly than the model predicts, observed < model means they reject parity violators more sharply than the model's rule/interval mixture with guessing allows.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(ex)).sum(axis=1)
    par = np.where(np.isnan(ex), np.nan, np.mod(ex, 2))
    s = np.nansum(par, axis=1)
    all_odd = (s == n)
    all_even = (s == 0)
    tp = df["target"].astype(int).values % 2
    mask = (n >= 2) & ((all_odd & (tp == 0)) | (all_even & (tp == 1)))
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())

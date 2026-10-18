# name: opposite_parity_yes_rate
# description: Yes rate for targets whose parity differs from every example (zero final-digit/parity similarity, the model's floor) on sets with >=2 examples; observed above null means people accept parity-violating targets more than the digit model's floor allows (parity/digit is not the operative dimension), below null means they reject them more sharply.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    same_par = np.sum(valid & ((np.where(valid, ex, 0) % 2) == (t[:, None] % 2)), axis=1)
    m = (n >= 2) & (same_par == 0)
    r = df["rating"].to_numpy().astype(float)
    return float(r[m].mean()) if m.sum() > 0 else 0.0

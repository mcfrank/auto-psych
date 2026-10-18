# name: factor_rule_nonmultiple_rejection
# description: In multi-example sets whose common factor g is >=3 and not a multiple of 5 (the model's common-factor rule), yes-rate on targets that are NOT multiples of g and share no example's final digit; observed < null means people reject rule-violating targets more firmly than the model (which only lacks a boost for them); observed > null means the model over-rejects them.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    Xi = np.nan_to_num(X).astype(int)
    g = np.gcd.reduce(Xi, axis=1)
    n = valid.sum(1)
    same_digit = (valid & (Xi % 10 == (t % 10)[:, None])).any(1)
    m = (n >= 2) & (g >= 3) & (g % 5 != 0) & (t % np.maximum(g, 1) != 0) & ~same_digit
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())

# name: shared_last_digit_set_nonmember_yes
# description: Yes rate for non-example targets whose last digit differs from the common last digit, among sets of >=2 examples that all share one last digit; observed below null means people reject non-members of an "ends in d" concept more sharply than the model's single tempered size principle allows (above null means less sharply).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = ex.notna().sum(axis=1)
    ld = ex % 10
    shared = (ld.nunique(axis=1) == 1) & (n >= 2)
    d = ld.iloc[:, 0]
    inset = ex.eq(t, axis=0).any(axis=1)
    mask = shared & (~inset) & ((t % 10) != d)
    return float(r[mask].mean()) if mask.sum() > 0 else 0.0

# name: mixed_parity_set_yes_rate
# description: Yes rate for non-example targets after 3+ example sets containing both odd and even numbers (no parity rule fits); observed > null means people generalise more broadly from rule-less, scattered sets than the model's interval/rule mixture, observed < null means more narrowly.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    par = ex % 2
    hasodd = par.eq(1).any(axis=1)
    haseven = par.eq(0).any(axis=1)
    inset = ex.eq(t, axis=0).any(axis=1)
    m = (n >= 3) & hasodd & haseven & ~inset
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float)[m].mean())

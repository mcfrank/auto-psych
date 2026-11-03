# name: shared_last_digit_target_yes_rate
# description: Among sets of >=2 examples that all share a last digit, the yes rate for non-example targets with that same last digit (0.5 if none); observed above/below null means the model under/over-generalises the "ends in d" rule.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1).to_numpy()
    last = (ex % 10).to_numpy()
    first = last[:, 0]
    same = np.all((last == first[:, None]) | np.isnan(last), axis=1) & (n >= 2)
    tgt = df["target"].astype(int).to_numpy()
    in_set = (ex.to_numpy() == tgt[:, None]).any(axis=1)
    m = same & (tgt % 10 == first) & ~in_set
    if m.sum() == 0:
        return 0.5
    return float(df["rating"].to_numpy()[m].mean())

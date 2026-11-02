# name: round_target_yes_in_nonround_sets
# description: "Yes" rate for round targets (multiples of 10, including 100) in sets containing no multiple of 10; observed above null = a round-number acceptance bias the model under-produces, below = model over-produces "yes" to round numbers (e.g. via ranges ending at 100).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    noround = ~(ex % 10 == 0).any(axis=1)
    t = df["target"].astype(int)
    m = noround & (t % 10 == 0)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())

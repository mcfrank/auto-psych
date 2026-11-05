# name: shared_last_digit_nonmatching_rate
# description: Yes rate for targets whose last digit differs from the common last digit of 3+-example sets whose examples all share one last digit (72,92,82,42 -> 25); observed below null means people reject non-matching numbers (even inside the range) more sharply than the model, whose broad-range prior lets intervals leak (above = less sharply).
def test_statistic(df):
    sets = df["number_set"].unique()
    keep = {}
    for s in sets:
        xs = [int(v) for v in str(s).split(",")]
        if len(xs) >= 3 and len({v % 10 for v in xs}) == 1:
            keep[s] = xs[0] % 10
    d = df[df["number_set"].isin(list(keep))]
    if len(d) == 0:
        return 0.0
    digit = d["number_set"].map(keep)
    m = (d["target"].astype(int) % 10) != digit
    return float(d.loc[m, "rating"].mean()) if m.sum() else 0.0

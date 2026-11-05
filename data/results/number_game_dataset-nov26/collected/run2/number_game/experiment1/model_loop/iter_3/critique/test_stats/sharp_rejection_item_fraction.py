# name: sharp_rejection_item_fraction
# description: Fraction of distinct (set, target) items whose mean yes-rate is <= 1/9 (near-unanimous rejection); observed > null means the model's graded overlap is too soft and under-produces the all-or-none rule-based rejections people make (observed < null: it over-produces them).
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean().values
    if len(m) == 0:
        return 0.0
    return float((m <= 1.0 / 9 + 1e-9).mean())

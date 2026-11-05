# name: item_rate_sd
# description: Standard deviation, across distinct (set, target) items, of each item's yes-rate; observed > null means items are more polarized (people agree more strongly on clear yes / clear no items) than the model's item probabilities produce, i.e. the model under-discriminates between items; observed < null means it over-discriminates.
def test_statistic(df):
    m = df.groupby(["number_set", "target"], sort=False)["rating"].mean()
    if len(m) < 2:
        return 0.0
    return float(m.std())

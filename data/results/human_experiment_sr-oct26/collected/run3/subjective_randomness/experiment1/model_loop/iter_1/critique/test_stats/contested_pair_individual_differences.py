# name: contested_pair_individual_differences
# description: Per-participant, the SD (across people) of agreement with the data-majority on the 25% of pairs with the weakest consensus; observed above null_mean means individual differences in WHICH sequence looks random (different criteria, not just different decisiveness) exceed what a single shared score with person-specific beta produces.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first_is_a = a < b
    key = np.where(first_is_a, a, b) + "|" + np.where(first_is_a, b, a)
    y = df["chose_left"].values
    cf = np.where(first_is_a, y, 1 - y)
    pm_ = pd.Series(cf).groupby(key).transform("mean").values
    ext = np.abs(pm_ - 0.5)
    pair_ext = pd.Series(ext).groupby(key).first()
    thr = np.quantile(pair_ext.values, 0.25)
    mask = ext <= thr
    agree = (cf == (pm_ >= 0.5)).astype(float)
    g = pd.Series(agree[mask]).groupby(df["participant_id"].values[mask]).mean()
    return float(g.std(ddof=0)) if len(g) > 1 else 0.0

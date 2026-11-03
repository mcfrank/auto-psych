# name: far_target_yes_rate
# description: Yes rate for targets more than 25 away from every example in the set; t_observed above null_mean means people give more distant/rule-based (or noise) yeses than the model predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    codes, uniq = pd.factorize(sets)
    tgt = df["target"].astype(int).values
    d = np.empty(len(df))
    for i, k in enumerate(uniq):
        a = np.array([int(x) for x in k.split(",")])
        idx = codes == i
        d[idx] = np.abs(tgt[idx][:, None] - a[None, :]).min(axis=1)
    m = d > 25
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

# name: last_digit_share_yes_rate
# description: Yes rate for non-example targets sharing the last digit with at least one but not all examples (multi-example sets); t_observed above null_mean means people use partial surface-digit similarity the model's all-or-none rules ignore.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    codes, uniq = pd.factorize(sets)
    tgt = df["target"].astype(int).values
    m = np.zeros(len(df), dtype=bool)
    for i, k in enumerate(uniq):
        a = np.array([int(x) for x in k.split(",")])
        if len(a) < 2:
            continue
        idx = np.where(codes == i)[0]
        t = tgt[idx]
        sh = (t[:, None] % 10 == a[None, :] % 10)
        cnt = sh.sum(axis=1)
        notex = ~(t[:, None] == a[None, :]).any(axis=1)
        m[idx] = (cnt >= 1) & (cnt < len(a)) & notex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

# name: outlier_example_yes_rate
# description: Yes-rate when the target is itself one of the shown examples AND the other >=3 examples all share a parity or a last digit that this example lacks (an odd-one-out example, e.g. 100 in 31,19,49,100); observed < null means people discard odd-one-out examples as noise more than the model, which nearly always keeps every example inside the concept.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pairs = pd.Series(list(zip(s, t))).drop_duplicates()
    flag = {}
    for ss, tt in pairs:
        a = [int(x) for x in ss.split(",")]
        if tt not in a:
            flag[(ss, tt)] = False
            continue
        rest = np.array([x for x in a if x != tt])
        f = False
        if len(rest) >= 3:
            if len(set(rest % 2)) == 1 and rest[0] % 2 != tt % 2:
                f = True
            if len(set(rest % 10)) == 1 and rest[0] % 10 != tt % 10:
                f = True
        flag[(ss, tt)] = f
    m = np.array([flag[(ss, tt)] for ss, tt in zip(s, t)])
    return float(y[m].mean()) if m.any() else 0.0

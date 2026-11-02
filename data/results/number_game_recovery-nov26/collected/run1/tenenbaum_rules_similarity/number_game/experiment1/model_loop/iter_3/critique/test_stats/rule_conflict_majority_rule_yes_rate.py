# name: rule_conflict_majority_rule_yes_rate
# description: For sets of >=3 examples whose examples do NOT all share a parity, yes-rate on non-example targets having the parity shared by all but one example (e.g. odd targets for 31,19,49,100); observed > null means people generalize by the majority rule, tolerating one exception, more than the model, which falls back to intervals when no rule covers every example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    codes, uniq = pd.factorize(s)
    arrs = [np.array([int(x) for x in u.split(",")]) for u in uniq]
    maj = np.full(len(arrs), -1)
    for i, a in enumerate(arrs):
        if len(a) >= 3:
            odd = int((a % 2).sum())
            if odd == len(a) - 1:
                maj[i] = 1
            elif odd == 1:
                maj[i] = 0
    mp = maj[codes]
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ins = np.array([str(tt) in u.split(",") for tt, u in zip(t, s)])
    m = (mp >= 0) & ~ins & (t % 2 == mp)
    return float(y[m].mean()) if m.any() else 0.0

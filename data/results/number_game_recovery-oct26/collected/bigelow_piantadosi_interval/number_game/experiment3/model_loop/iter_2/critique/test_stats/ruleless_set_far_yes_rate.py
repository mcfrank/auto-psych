# name: ruleless_set_far_yes_rate
# description: For 3-4 example sets that no listed rule of at most 50 members contains entirely (e.g. 92,82,27,42; 79,47,62,98), yes rate on non-example targets more than 10 from every example; observed above null_mean means people say yes to unrelated distant numbers after unstructured sets more than the model predicts (model too conservative), below means less.
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [[2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81], [10, 100]]
    rules += [list(N[N % k == 0]) for k in range(2, 13)]
    rules += [list(N[N % 10 == d]) for d in range(0, 10)]
    rules += [[i for i in N if str(d) in str(i)] for d in range(1, 10)]
    rules.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append(list(N[N % 2 == 1]))
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    M = np.zeros((len(rules), 101), bool)
    for i, c in enumerate(rules):
        M[i, c] = True
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    ok = {}
    for s in uniq:
        e = [int(x) for x in s.split(",")]
        ok[s] = len(e) >= 3 and not M[:, e].all(axis=1).any()
    sel_set = sets.map(ok).to_numpy(dtype=bool)
    ex = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    sel = sel_set & (dist > 10)
    return float(df["rating"].to_numpy()[sel].mean()) if sel.any() else 0.0

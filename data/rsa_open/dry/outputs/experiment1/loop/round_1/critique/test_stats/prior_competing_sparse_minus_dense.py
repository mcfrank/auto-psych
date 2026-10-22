# name: prior_competing_sparse_minus_dense
# description: On prior trials with duplicate 0-feature distractors and two singletons (sparse 1-feature vs dense all-features), the mean points allocated to the sparse singleton minus the dense singleton.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if not pd.isna(sc["heard"]):
            continue
        objs = json.loads(sc["objects"])
        counts = {tuple(o): objs.count(o) for o in objs}
        singletons = [i for i, o in enumerate(objs) if counts[tuple(o)] == 1]
        duplicates = [i for i, o in enumerate(objs) if counts[tuple(o)] > 1]
        if len(singletons) == 2 and duplicates and all(sum(objs[i]) == 0 for i in duplicates):
            fc = [sum(objs[i]) for i in singletons]
            s_idx = singletons[int(np.argmin(fc))]
            d_idx = singletons[int(np.argmax(fc))]
            table[sc["objects"]] = (s_idx, d_idx)
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if not pd.isna(h):
            continue
        key = df["objects"].iloc[idx]
        if key in table:
            s_idx, d_idx = table[key]
            total += (p[s_idx] - p[d_idx])
            count += 1
    return float(total / max(count, 1))

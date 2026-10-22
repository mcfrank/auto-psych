# name: prior_all_singleton_full_complexity_points
# description: On prior trials where all objects are distinct singletons with ascending feature counts from zero to all features, the mean points allocated to the all-features object.
def test_statistic(df):
    table = {}
    distinct = df[["objects", "heard"]].drop_duplicates()
    for _, sc in distinct.iterrows():
        if not pd.isna(sc["heard"]):
            continue
        objs = json.loads(sc["objects"])
        if len(set(tuple(o) for o in objs)) == len(objs):
            n_feats = len(objs[0])
            full = [i for i, o in enumerate(objs) if sum(o) == n_feats]
            if full and len(objs) > 2:
                table[sc["objects"]] = full[0]
    pts_list = [json.loads(p) if isinstance(p, str) else p for p in df["points"]]
    total, count = 0.0, 0
    for idx, (h, p) in enumerate(zip(df["heard"], pts_list)):
        if not pd.isna(h):
            continue
        key = df["objects"].iloc[idx]
        if key in table:
            total += p[table[key]]
            count += 1
    return float(total / max(count, 1))

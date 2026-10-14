# name: prior_max_feature_rate_all_singletons
# description: Rate of choosing the maximal-feature object on mumble trials where all objects in the display are visual singletons.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        obj_tuples = [tuple(o) for o in objs]
        if all(obj_tuples.count(t) == 1 for t in obj_tuples):
            counts = [sum(o) for o in objs]
            max_c = max(counts)
            max_idxs = [i for i, c in enumerate(counts) if c == max_c]
            if len(max_idxs) == 1 and max_c > 0:
                lookup[objs_str] = max_idxs[0]
    keys = sub['objects'].tolist()
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask])) if np.any(mask) else 0.0

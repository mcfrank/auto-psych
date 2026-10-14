# name: prior_dup_full_rate_with_empty_singleton
# description: Rate of choosing the full duplicated object on mumble trials where the display contains an empty singleton object and a full object with duplicate copies.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        counts = [sum(o) for o in objs]
        obj_tuples = [tuple(o) for o in objs]
        min_c, max_c = min(counts), max(counts)
        if min_c == 0 and max_c > 1:
            empty_sing = [i for i, c in enumerate(counts) if c == 0 and obj_tuples.count(obj_tuples[i]) == 1]
            full_dup = [i for i, c in enumerate(counts) if c == max_c and obj_tuples.count(obj_tuples[i]) > 1]
            if len(empty_sing) == 1 and len(full_dup) >= 2:
                lookup[objs_str] = full_dup[0]
    keys = sub['objects'].tolist()
    targets = np.array([lookup.get(k, -1) for k in keys])
    mask = targets != -1
    return float(np.mean(sub['choice'].to_numpy()[mask] == targets[mask])) if np.any(mask) else 0.0

# name: prior_empty_singleton_rate_dup_full
# description: Rate of choosing the empty singleton object on mumble trials where the empty object has zero features and the full object has duplicate copies.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        counts = [sum(o) for o in objs]
        obj_tuples = [tuple(o) for o in objs]
        singletons = {i for i, t in enumerate(obj_tuples) if obj_tuples.count(t) == 1}
        max_c = max(counts)
        if min(counts) == 0 and max_c > 0:
            empty_singletons = {i for i, c in enumerate(counts) if c == 0 and i in singletons}
            if len(empty_singletons) == 1 and counts.count(max_c) > 1:
                lookup[objs_str] = list(empty_singletons)[0]
    keys = sub['objects'].tolist()
    is_valid = [k in lookup for k in keys]
    if not any(is_valid):
        return 0.0
    sub_filt = sub[is_valid]
    valid_keys = [k for k in keys if k in lookup]
    choices = sub_filt['choice'].to_numpy()
    targets = np.array([lookup[k] for k in valid_keys])
    return float(np.mean(choices == targets))

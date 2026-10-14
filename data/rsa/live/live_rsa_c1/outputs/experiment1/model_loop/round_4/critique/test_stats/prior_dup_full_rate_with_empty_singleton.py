# name: prior_dup_full_rate_with_empty_singleton
# description: Rate of choosing the full duplicated object on mumble trials where the display contains an empty singleton object and a full object with duplicate copies.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        counts = [sum(o) for o in objs]
        min_c = min(counts)
        max_c = max(counts)
        if min_c == 0 and max_c > 1:
            empty_idxs = [i for i, c in enumerate(counts) if c == 0]
            max_idxs = {i for i, c in enumerate(counts) if c == max_c}
            obj_tuples = [tuple(o) for o in objs]
            if len(empty_idxs) == 1 and obj_tuples.count(obj_tuples[empty_idxs[0]]) == 1 and len(max_idxs) > 1 and obj_tuples.count(obj_tuples[list(max_idxs)[0]]) > 1:
                lookup[objs_str] = max_idxs
    keys = sub['objects'].tolist()
    is_valid = [k in lookup for k in keys]
    if not any(is_valid):
        return 0.0
    sub_filt = sub[is_valid]
    valid_keys = [k for k in keys if k in lookup]
    choices = sub_filt['choice'].to_numpy()
    hits = [int(c) in lookup[k] for c, k in zip(choices, valid_keys)]
    return float(np.mean(hits))

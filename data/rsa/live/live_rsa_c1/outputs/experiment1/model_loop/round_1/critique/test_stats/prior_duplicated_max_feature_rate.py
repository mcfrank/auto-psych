# name: prior_duplicated_max_feature_rate
# description: Rate of choosing the maximal-feature class on mumble trials where the most complex object has multiple identical copies in the display.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        counts = [sum(o) for o in objs]
        max_c = max(counts)
        max_idxs = [i for i, c in enumerate(counts) if c == max_c]
        obj_tuples = [tuple(o) for o in objs]
        if len(max_idxs) > 1 and obj_tuples.count(obj_tuples[max_idxs[0]]) > 1:
            lookup[objs_str] = max_idxs[0]
    keys = sub['objects'].tolist()
    is_valid = [k in lookup for k in keys]
    if not any(is_valid):
        return 0.0
    sub_filt = sub[is_valid]
    valid_keys = [k for k in keys if k in lookup]
    choices = sub_filt['choice'].to_numpy()
    targets = np.array([lookup[k] for k in valid_keys])
    return float(np.mean(choices == targets))

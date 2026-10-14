# name: prior_max_feature_rate
# description: Rate of choosing the maximal-feature object on mumble trials where objects vary in total feature count.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        counts = [sum(o) for o in objs]
        max_c = max(counts)
        max_idxs = {i for i, c in enumerate(counts) if c == max_c}
        if len(max_idxs) < len(objs):
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

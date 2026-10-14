# name: prior_intermediate_features_rate
# description: Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects.
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
            inter = {i for i, c in enumerate(counts) if 0 < c < max_c}
            if len(inter) > 0:
                lookup[objs_str] = inter
    keys = sub['objects'].tolist()
    is_valid = [k in lookup for k in keys]
    if not any(is_valid):
        return 0.0
    sub_filt = sub[is_valid]
    valid_keys = [k for k in keys if k in lookup]
    choices = sub_filt['choice'].to_numpy()
    hits = [int(c) in lookup[k] for c, k in zip(choices, valid_keys)]
    return float(np.mean(hits))

# name: prior_singleton_rate_with_duplicates
# description: Rate of choosing any visual singleton object on mumble trials where the display contains both singleton and duplicated objects.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        obj_tuples = [tuple(o) for o in objs]
        singletons = {i for i, t in enumerate(obj_tuples) if obj_tuples.count(t) == 1}
        if 0 < len(singletons) < len(objs):
            lookup[objs_str] = singletons
    keys = sub['objects'].tolist()
    choices = sub['choice'].to_numpy()
    hits = [int(c) in lookup[k] for c, k in zip(choices, keys) if k in lookup]
    return float(np.mean(hits)) if hits else 0.0

# name: prior_intermediate_features_rate
# description: Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects.
def test_statistic(df):
    sub = df[df['query'] == 'prior']
    uniques = sub['objects'].drop_duplicates()
    lookup = {}
    for objs_str in uniques:
        objs = json.loads(objs_str)
        counts = [sum(o) for o in objs]
        min_c, max_c = min(counts), max(counts)
        if min_c == 0 and max_c > 1:
            inter = {i for i, c in enumerate(counts) if 0 < c < max_c}
            if inter:
                lookup[objs_str] = inter
    keys = sub['objects'].tolist()
    choices = sub['choice'].to_numpy()
    hits = [c in lookup[k] for c, k in zip(choices, keys) if k in lookup]
    return float(np.mean(hits)) if hits else 0.0

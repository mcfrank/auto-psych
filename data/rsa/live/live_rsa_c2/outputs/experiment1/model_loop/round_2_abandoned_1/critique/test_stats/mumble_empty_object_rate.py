# name: mumble_empty_object_rate
# description: Proportion of choices for the featureless object among four-object mumble trials containing at least one featureless object.
def test_statistic(df):
    unique_objs = df[df['query'] == 'prior']['objects'].drop_duplicates()
    empty_map = {}
    for s in unique_objs:
        objs = json.loads(s)
        if len(objs) == 4 and any(sum(o) == 0 for o in objs):
            first_empty = [objs.index(o) for o in objs if sum(o) == 0][0]
            empty_map[s] = first_empty
    sub_df = df[(df['query'] == 'prior') & (df['objects'].isin(empty_map))]
    if len(sub_df) == 0:
        return 0.0
    targets = np.array([empty_map[o] for o in sub_df['objects']])
    return float(np.mean(sub_df['choice'].to_numpy() == targets))

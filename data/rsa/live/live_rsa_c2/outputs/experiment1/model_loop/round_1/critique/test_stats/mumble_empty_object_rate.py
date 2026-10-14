# name: mumble_empty_object_rate
# description: Proportion of choices for the featureless object [0, 0, 0, 0] among four-object mumble trials containing at least one featureless object.
def test_statistic(df):
    unique_objs = df.loc[df['query'] == 'prior', 'objects'].unique()
    empty_map = {}
    for obj_str in unique_objs:
        objs = json.loads(obj_str)
        if len(objs) == 4:
            for i, o in enumerate(objs):
                if sum(o) == 0:
                    empty_map[obj_str] = objs.index(o)
                    break
    prior_rows = df[df['query'] == 'prior']
    targets = [empty_map.get(o, -1) for o in prior_rows['objects']]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = prior_rows[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))

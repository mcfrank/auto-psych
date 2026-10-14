# name: mumble_pairwise_rich_object_rate
# description: Proportion of choices for the object with more features among two-object mumble trials with unequal feature counts.
def test_statistic(df):
    unique_objs = df.loc[df['query'] == 'prior', 'objects'].unique()
    full_map = {}
    for obj_str in unique_objs:
        objs = json.loads(obj_str)
        if len(objs) == 2:
            cnt0, cnt1 = sum(objs[0]), sum(objs[1])
            if cnt0 != cnt1:
                full_map[obj_str] = 0 if cnt0 > cnt1 else 1
    prior_rows = df[df['query'] == 'prior']
    targets = [full_map.get(o, -1) for o in prior_rows['objects']]
    mask = [t != -1 for t in targets]
    if not any(mask):
        return 0.0
    sub_df = prior_rows[mask]
    sub_targets = np.array([t for t in targets if t != -1])
    return float(np.mean(sub_df['choice'].to_numpy() == sub_targets))

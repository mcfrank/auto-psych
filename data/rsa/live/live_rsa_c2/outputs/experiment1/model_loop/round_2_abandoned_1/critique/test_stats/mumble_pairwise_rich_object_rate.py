# name: mumble_pairwise_rich_object_rate
# description: Proportion of choices for the object with more features among two-object mumble trials with unequal feature counts.
def test_statistic(df):
    unique_objs = df[df['query'] == 'prior']['objects'].drop_duplicates()
    rich_map = {}
    for s in unique_objs:
        objs = json.loads(s)
        if len(objs) == 2:
            cnt0, cnt1 = sum(objs[0]), sum(objs[1])
            if cnt0 != cnt1:
                rich_map[s] = 0 if cnt0 > cnt1 else 1
    sub_df = df[(df['query'] == 'prior') & (df['objects'].isin(rich_map))]
    if len(sub_df) == 0:
        return 0.0
    targets = np.array([rich_map[o] for o in sub_df['objects']])
    return float(np.mean(sub_df['choice'].to_numpy() == targets))

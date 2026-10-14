# name: mumble_intermediate_feature_rate
# description: Proportion of choices for objects with strictly intermediate feature counts among mumble trials where the display contains objects of at least three distinct feature counts.
def test_statistic(df):
    unique_objs = df[df['query'] == 'prior']['objects'].drop_duplicates()
    inter_map = {}
    for s in unique_objs:
        objs = json.loads(s)
        counts = [sum(o) for o in objs]
        unique_counts = sorted(list(set(counts)))
        if len(unique_counts) >= 3:
            intermediate = set(unique_counts[1:-1])
            classes = set(objs.index(o) for o in objs if sum(o) in intermediate)
            inter_map[s] = classes
    sub_df = df[(df['query'] == 'prior') & (df['objects'].isin(inter_map))]
    if len(sub_df) == 0:
        return 0.0
    is_inter = [c in inter_map[o] for o, c in zip(sub_df['objects'], sub_df['choice'])]
    return float(np.mean(is_inter))

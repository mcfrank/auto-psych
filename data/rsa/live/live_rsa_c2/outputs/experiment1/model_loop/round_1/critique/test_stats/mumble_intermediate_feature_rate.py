# name: mumble_intermediate_feature_rate
# description: Proportion of choices for objects with strictly intermediate feature counts among mumble trials where the display contains objects of at least three distinct feature counts.
def test_statistic(df):
    unique_objs = df.loc[df['query'] == 'prior', 'objects'].unique()
    inter_map = {}
    for obj_str in unique_objs:
        objs = json.loads(obj_str)
        seen = {}
        for i, o in enumerate(objs):
            t = tuple(o)
            if t not in seen:
                seen[t] = (i, sum(o))
        counts = [cnt for idx, cnt in seen.values()]
        unique_counts = sorted(set(counts))
        if len(unique_counts) >= 3:
            min_c, max_c = unique_counts[0], unique_counts[-1]
            inter_indices = {idx for idx, cnt in seen.values() if min_c < cnt < max_c}
            inter_map[obj_str] = inter_indices
    prior_rows = df[df['query'] == 'prior']
    matches = [inter_map.get(o, set()) for o in prior_rows['objects']]
    mask = [len(s) > 0 for s in matches]
    if not any(mask):
        return 0.0
    sub_df = prior_rows[mask]
    sub_targets = [matches[i] for i, m in enumerate(mask) if m]
    chosen = [c in s for c, s in zip(sub_df['choice'], sub_targets)]
    return float(np.mean(chosen))

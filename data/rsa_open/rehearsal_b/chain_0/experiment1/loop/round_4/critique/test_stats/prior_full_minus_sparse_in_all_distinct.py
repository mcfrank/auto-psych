# name: prior_full_minus_sparse_in_all_distinct
# description: Mean point difference between the full-feature object and sparse objects in 3-object prior trials where all objects are distinct.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if not pd.isna(sc['heard']):
            continue
        objs = json.loads(sc['objects'])
        if len(objs) == 3 and len(set(tuple(o) for o in objs)) == 3:
            lens = sorted([sum(o) for o in objs])
            if lens == [1, 2, 3]:
                idx_full = [i for i, o in enumerate(objs) if sum(o) == 3][0]
                idx_sparse = [i for i, o in enumerate(objs) if sum(o) < 3]
                target_scenes[(sc['objects'], sc['heard'])] = (idx_full, idx_sparse)
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), (i_full, i_sparse) in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'].isna())]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += (pts[i_full] - np.mean([pts[j] for j in i_sparse]))
            cnt += 1
    return float(tot / max(cnt, 1))

# name: prior_points_empty_object_with_partial_duplicates
# description: Mean points allocated to the zero-feature empty object in prior trials containing duplicate 1-feature objects and a 2-feature singleton.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if not pd.isna(sc['heard']):
            continue
        objs = json.loads(sc['objects'])
        empties = [i for i, o in enumerate(objs) if sum(o) == 0]
        dups_1 = [i for i, o in enumerate(objs) if sum(o) == 1 and objs.count(o) == 2]
        sings_2 = [i for i, o in enumerate(objs) if sum(o) == 2 and objs.count(o) == 1]
        if len(empties) == 1 and len(dups_1) == 2 and len(sings_2) == 1:
            target_scenes[(sc['objects'], sc['heard'])] = empties[0]
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), idx in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'].isna())]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += pts[idx]
            cnt += 1
    return float(tot / max(cnt, 1))

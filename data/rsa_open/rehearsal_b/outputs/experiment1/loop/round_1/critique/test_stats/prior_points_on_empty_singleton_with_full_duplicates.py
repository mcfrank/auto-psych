# name: prior_points_on_empty_singleton_with_full_duplicates
# description: Mean points allocated to the empty zero-feature singleton object in prior trials where all other objects are identical full-feature duplicates.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if not pd.isna(sc['heard']):
            continue
        objs = json.loads(sc['objects'])
        empties = [i for i, o in enumerate(objs) if sum(o) == 0]
        fulls = [i for i, o in enumerate(objs) if sum(o) == len(o)]
        if len(empties) == 1 and len(fulls) == len(objs) - 1:
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

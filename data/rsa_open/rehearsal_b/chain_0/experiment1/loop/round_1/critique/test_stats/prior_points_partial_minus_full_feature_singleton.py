# name: prior_points_partial_minus_full_feature_singleton
# description: Mean point difference between partial-feature singletons and full-feature singletons in prior trials containing empty duplicate distractors.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if not pd.isna(sc['heard']):
            continue
        objs = json.loads(sc['objects'])
        empties = [i for i, o in enumerate(objs) if sum(o) == 0]
        partial = [i for i, o in enumerate(objs) if 1 <= sum(o) < len(o) and objs.count(o) == 1]
        full = [i for i, o in enumerate(objs) if sum(o) == len(o) and objs.count(o) == 1]
        if len(empties) >= 2 and len(partial) == 1 and len(full) == 1:
            target_scenes[(sc['objects'], sc['heard'])] = (partial[0], full[0])
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), (i_part, i_full) in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'].isna())]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += (pts[i_part] - pts[i_full])
            cnt += 1
    return float(tot / max(cnt, 1))

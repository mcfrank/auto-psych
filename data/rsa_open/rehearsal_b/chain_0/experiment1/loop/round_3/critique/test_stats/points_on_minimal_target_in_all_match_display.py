# name: points_on_minimal_target_in_all_match_display
# description: Mean points allocated to the unique 1-feature minimal target in 4-object informative scenes where all objects satisfy the heard word and competitors have 2 or 3 features.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        if len(objs) == 4 and all(o[h] == 1 for o in objs):
            min_sings = [i for i, o in enumerate(objs) if sum(o) == 1 and objs.count(o) == 1]
            mid_sings = [i for i, o in enumerate(objs) if sum(o) == 2 and objs.count(o) == 1]
            full_dups = [i for i, o in enumerate(objs) if sum(o) == 3 and objs.count(o) == 2]
            if len(min_sings) == 1 and len(mid_sings) == 1 and len(full_dups) == 2:
                target_scenes[(sc['objects'], sc['heard'])] = min_sings[0]
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), idx in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'] == h_val)]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += pts[idx]
            cnt += 1
    return float(tot / max(cnt, 1))

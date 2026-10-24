# name: points_on_scalar_competitor_with_duplicate_targets
# description: Mean points allocated to the 3-feature competitor in informative scalar implicature scenes where the heard word is shared with two 1-feature duplicate targets.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        comp = [i for i, o in enumerate(objs) if o[h] == 1]
        dups_1 = [i for i in comp if sum(objs[i]) == 1 and objs.count(objs[i]) == 2]
        full_3 = [i for i in comp if sum(objs[i]) == len(objs[i]) and objs.count(objs[i]) == 1]
        if len(dups_1) == 2 and len(full_3) == 1 and len(comp) == 3:
            target_scenes[(sc['objects'], sc['heard'])] = full_3[0]
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

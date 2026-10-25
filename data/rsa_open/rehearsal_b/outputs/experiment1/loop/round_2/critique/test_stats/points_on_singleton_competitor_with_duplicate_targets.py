# name: points_on_singleton_competitor_with_duplicate_targets
# description: Mean points allocated to a 2-feature singleton competitor in informative scenes where the heard word is also true of duplicate 1-feature minimal targets.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        comp = [i for i, o in enumerate(objs) if o[h] == 1]
        dups = [i for i in comp if objs.count(objs[i]) == 2 and sum(objs[i]) == 1]
        sings = [i for i in comp if objs.count(objs[i]) == 1 and sum(objs[i]) == 2]
        if len(dups) == 2 and len(sings) == 1:
            target_scenes[(sc['objects'], sc['heard'])] = sings[0]
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

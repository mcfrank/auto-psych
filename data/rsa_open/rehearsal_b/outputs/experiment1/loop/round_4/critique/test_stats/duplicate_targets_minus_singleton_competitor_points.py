# name: duplicate_targets_minus_singleton_competitor_points
# description: Total points allocated to duplicate 2-feature targets minus points allocated to a 3-feature singleton competitor in informative scalar implicature scenes where the heard word is true of all three.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        comp = [i for i, o in enumerate(objs) if o[h] == 1]
        dups = [i for i in comp if objs.count(objs[i]) == 2 and sum(objs[i]) == 2]
        sings = [i for i in comp if objs.count(objs[i]) == 1 and sum(objs[i]) == 3]
        if len(dups) == 2 and len(sings) == 1:
            target_scenes[(sc['objects'], sc['heard'])] = (dups, sings[0])
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), (d_idxs, s_idx) in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'] == h_val)]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += (sum(pts[i] for i in d_idxs) - pts[s_idx])
            cnt += 1
    return float(tot / max(cnt, 1))

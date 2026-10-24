# name: points_on_full_competitor_with_triplicate_targets
# description: Mean points allocated to the 3-feature singleton competitor in 4-object informative scenes where the heard word is shared with three 1-feature triplicate targets.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        if len(objs) == 4 and all(o[h] == 1 for o in objs):
            trips = [i for i, o in enumerate(objs) if sum(o) == 1 and objs.count(o) == 3]
            fulls = [i for i, o in enumerate(objs) if sum(o) == len(o) and objs.count(o) == 1]
            if len(trips) == 3 and len(fulls) == 1:
                target_scenes[(sc['objects'], sc['heard'])] = fulls[0]
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

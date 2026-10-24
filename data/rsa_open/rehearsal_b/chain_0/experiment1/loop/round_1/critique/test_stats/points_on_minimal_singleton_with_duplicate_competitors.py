# name: points_on_minimal_singleton_with_duplicate_competitors
# description: Mean points allocated to a 1-feature minimal singleton in informative scenes where the heard word is true of all objects and competitors are multi-feature duplicates.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        if all(o[h] == 1 for o in objs):
            min_sings = [i for i, o in enumerate(objs) if sum(o) == 1 and objs.count(o) == 1]
            dup_comps = [i for i, o in enumerate(objs) if sum(o) > 1 and objs.count(o) > 1]
            if len(min_sings) == 1 and len(dup_comps) >= 2:
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

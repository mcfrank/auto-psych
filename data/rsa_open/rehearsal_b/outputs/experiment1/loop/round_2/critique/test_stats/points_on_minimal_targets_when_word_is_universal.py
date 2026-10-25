# name: points_on_minimal_targets_when_word_is_universal
# description: Total points allocated to 1-feature duplicate targets in informative scenes where the heard word is true of all objects and competitors have multiple features.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        if all(o[h] == 1 for o in objs):
            mins = [i for i, o in enumerate(objs) if sum(o) == 1 and objs.count(o) > 1]
            multis = [i for i, o in enumerate(objs) if sum(o) > 1 and objs.count(o) == 1]
            if len(mins) == 2 and len(multis) == 2:
                target_scenes[(sc['objects'], sc['heard'])] = mins
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), idxs in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'] == h_val)]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += sum(pts[i] for i in idxs)
            cnt += 1
    return float(tot / max(cnt, 1))

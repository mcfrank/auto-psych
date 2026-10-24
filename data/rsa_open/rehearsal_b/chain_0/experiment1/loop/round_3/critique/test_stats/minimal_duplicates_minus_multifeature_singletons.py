# name: minimal_duplicates_minus_multifeature_singletons
# description: Total points on duplicate 1-feature targets minus total points on 2-feature singleton competitors in universal scenes where all four objects satisfy the heard word.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        if all(o[h] == 1 for o in objs):
            mins = [i for i, o in enumerate(objs) if sum(o) == 1 and objs.count(o) == 2]
            twos = [i for i, o in enumerate(objs) if sum(o) == 2 and objs.count(o) == 1]
            if len(mins) == 2 and len(twos) == 2:
                target_scenes[(sc['objects'], sc['heard'])] = (mins, twos)
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), (mins, twos) in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'] == h_val)]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += (sum(pts[i] for i in mins) - sum(pts[j] for j in twos))
            cnt += 1
    return float(tot / max(cnt, 1))

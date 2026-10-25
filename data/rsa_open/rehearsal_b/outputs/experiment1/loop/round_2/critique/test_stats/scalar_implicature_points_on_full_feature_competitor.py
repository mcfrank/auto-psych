# name: scalar_implicature_points_on_full_feature_competitor
# description: Mean points allocated to the 3-feature competitor object in informative scalar implicature scenes where the heard word is shared between a 1-feature target and a 3-feature competitor.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if pd.isna(sc['heard']):
            continue
        h = int(sc['heard'])
        objs = json.loads(sc['objects'])
        comp = [i for i, o in enumerate(objs) if o[h] == 1]
        non_comp = [i for i, o in enumerate(objs) if o[h] == 0]
        if len(comp) == 2 and len(non_comp) >= 2:
            comp_lens = sorted([sum(objs[i]) for i in comp])
            if comp_lens == [1, 3]:
                nc_objs = [objs[i] for i in non_comp]
                if len(set(tuple(x) for x in nc_objs)) == 1:
                    idx_3 = [i for i in comp if sum(objs[i]) == 3][0]
                    target_scenes[(sc['objects'], sc['heard'])] = idx_3
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

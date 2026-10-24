# name: prior_empty_minus_partial_singleton_with_full_duplicates
# description: Mean point difference between an empty zero-feature singleton and a 2-feature partial singleton in prior trials containing duplicate full-feature objects.
def test_statistic(df):
    scenes = df[['objects', 'heard']].drop_duplicates()
    target_scenes = {}
    for _, sc in scenes.iterrows():
        if not pd.isna(sc['heard']):
            continue
        objs = json.loads(sc['objects'])
        empties = [i for i, o in enumerate(objs) if sum(o) == 0 and objs.count(o) == 1]
        twos = [i for i, o in enumerate(objs) if sum(o) == 2 and objs.count(o) == 1]
        full_dups = [i for i, o in enumerate(objs) if sum(o) == len(o) and objs.count(o) == 2]
        if len(empties) == 1 and len(twos) == 1 and len(full_dups) == 2 and len(objs) == 4:
            target_scenes[(sc['objects'], sc['heard'])] = (empties[0], twos[0])
    if not target_scenes:
        return 0.0
    tot, cnt = 0.0, 0
    for (obj_str, h_val), (i_emp, i_two) in target_scenes.items():
        sub_pts = df[(df['objects'] == obj_str) & (df['heard'].isna())]['points']
        for p_str in sub_pts:
            pts = json.loads(p_str) if isinstance(p_str, str) else p_str
            tot += (pts[i_emp] - pts[i_two])
            cnt += 1
    return float(tot / max(cnt, 1))

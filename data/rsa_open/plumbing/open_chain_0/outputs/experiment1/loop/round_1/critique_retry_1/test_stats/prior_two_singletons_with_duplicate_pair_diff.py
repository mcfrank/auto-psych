# name: prior_two_singletons_with_duplicate_pair_diff
# description: Mean point difference between the sparser singleton and the richer singleton on prior trials (no word heard) that contain two distinct singletons alongside an identical pair of duplicate objects, testing whether the prior's distinctiveness and feature-count weights distort preference between the two singletons.
def test_statistic(df):
    prior_mask = df["heard"].isna()
    sub = df[prior_mask]
    if len(sub) == 0:
        return 0.0
    scenes = sub["objects"].unique()
    scene_map = {}
    for objs_str in scenes:
        objs = json.loads(objs_str)
        dup_counts = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        sings = [i for i, c in enumerate(dup_counts) if c == 1]
        dups = [i for i, c in enumerate(dup_counts) if c == 2]
        if len(sings) == 2 and len(dups) == 2:
            s0, s1 = sings[0], sings[1]
            if sum(objs[s0]) <= sum(objs[s1]):
                scene_map[objs_str] = (s0, s1)
            else:
                scene_map[objs_str] = (s1, s0)
    matched = [scene_map.get(s, None) for s in sub["objects"]]
    valid_idx = [i for i, m in enumerate(matched) if m is not None]
    if not valid_idx:
        return 0.0
    pts_series = sub["points"].iloc[valid_idx]
    unique_pts = pts_series.unique()
    pts_map = {s: json.loads(s) for s in unique_pts}
    vals = []
    for s, idx in zip(pts_series, valid_idx):
        p = pts_map[s]
        s_low, s_high = matched[idx]
        vals.append(p[s_low] - p[s_high])
    return float(np.mean(vals))

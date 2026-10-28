# name: prior_odd_one_out_rich_vs_sparse_asymmetry
# description: Difference on prior trials between the mean points given to a unique all-features singleton (against featureless duplicates) and the mean points given to a unique featureless singleton (against all-features duplicates), testing feature-count asymmetry when singleton and distinctiveness cues are symmetric.
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
        if len(sings) == 1 and len(objs) >= 3:
            s_idx = sings[0]
            other_idx = [i for i in range(len(objs)) if i != s_idx][0]
            f_sing = sum(objs[s_idx])
            f_other = sum(objs[other_idx])
            if f_sing > f_other and f_other == 0:
                scene_map[objs_str] = ("rich_sing", s_idx)
            elif f_sing == 0 and f_other > 0:
                scene_map[objs_str] = ("sparse_sing", s_idx)
    matched = [scene_map.get(s, None) for s in sub["objects"]]
    valid_idx = [i for i, m in enumerate(matched) if m is not None]
    if not valid_idx:
        return 0.0
    pts_series = sub["points"].iloc[valid_idx]
    unique_pts = pts_series.unique()
    pts_map = {s: json.loads(s) for s in unique_pts}
    rich_vals = []
    sparse_vals = []
    for s, idx in zip(pts_series, valid_idx):
        p = pts_map[s]
        kind, s_idx = matched[idx]
        if kind == "rich_sing":
            rich_vals.append(p[s_idx])
        else:
            sparse_vals.append(p[s_idx])
    if not rich_vals or not sparse_vals:
        return 0.0
    return float(np.mean(rich_vals) - np.mean(sparse_vals))

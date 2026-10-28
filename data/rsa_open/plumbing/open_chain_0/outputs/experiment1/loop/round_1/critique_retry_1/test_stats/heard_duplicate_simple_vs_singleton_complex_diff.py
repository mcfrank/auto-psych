# name: heard_duplicate_simple_vs_singleton_complex_diff
# description: Mean difference in points between the two identical simple true referents (summed) and the singleton complex true referent in heard-word scenes where the word applies to a duplicate pair of fewer-feature objects and a singleton object with more features.
def test_statistic(df):
    heard_mask = df["heard"].notna()
    sub = df[heard_mask]
    if len(sub) == 0:
        return 0.0
    scenes = sub[["objects", "heard"]].drop_duplicates()
    scene_info = {}
    for objs_str, h_val in zip(scenes["objects"], scenes["heard"]):
        objs = json.loads(objs_str)
        h = int(h_val)
        true_indices = [i for i, o in enumerate(objs) if o[h] == 1]
        dup_counts = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        dup_true = [i for i in true_indices if dup_counts[i] == 2]
        sing_true = [i for i in true_indices if dup_counts[i] == 1]
        if len(dup_true) == 2 and len(sing_true) >= 1:
            f_dup = sum(objs[dup_true[0]])
            complex_sing = [i for i in sing_true if sum(objs[i]) > f_dup]
            if len(complex_sing) >= 1:
                # pick the most feature-rich singleton true referent
                c_idx = max(complex_sing, key=lambda idx: sum(objs[idx]))
                scene_info[(objs_str, h)] = (dup_true[0], dup_true[1], c_idx)
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    matched = [scene_info.get(k, None) for k in keys]
    valid_idx = [i for i, m in enumerate(matched) if m is not None]
    if not valid_idx:
        return 0.0
    pts_series = sub["points"].iloc[valid_idx]
    unique_pts = pts_series.unique()
    pts_map = {s: json.loads(s) for s in unique_pts}
    vals = []
    for s, idx in zip(pts_series, valid_idx):
        p = pts_map[s]
        d0, d1, c = matched[idx]
        vals.append((p[d0] + p[d1]) - p[c])
    return float(np.mean(vals))

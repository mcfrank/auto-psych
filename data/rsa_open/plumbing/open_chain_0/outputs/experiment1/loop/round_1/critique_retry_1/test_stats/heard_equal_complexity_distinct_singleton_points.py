# name: heard_equal_complexity_distinct_singleton_points
# description: Mean points allocated to the most contextually distinctive singleton true referent in heard-word scenes where multiple singletons share the minimum feature count among true referents, probing whether speaker pragmatic choice versus listener prior over- or under-weights singleton distinctiveness.
def test_statistic(df):
    heard_mask = df["heard"].notna()
    sub = df[heard_mask]
    if len(sub) == 0:
        return 0.0
    scenes = sub[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for objs_str, h_val in zip(scenes["objects"], scenes["heard"]):
        objs = json.loads(objs_str)
        h = int(h_val)
        arr = np.array(objs, dtype=float)
        dup_counts = [sum(1 for o2 in objs if o2 == objs[i]) for i in range(len(objs))]
        diff = np.abs(arr[:, None, :] - arr[None, :, :]).sum(axis=(1, 2))
        true_indices = [i for i, o in enumerate(objs) if o[h] == 1]
        if not true_indices:
            continue
        min_f = min(sum(objs[i]) for i in true_indices)
        min_sing_true = [i for i in true_indices if sum(objs[i]) == min_f and dup_counts[i] == 1]
        if len(min_sing_true) >= 2:
            dists = [diff[i] for i in min_sing_true]
            if max(dists) > min(dists):
                best = max(min_sing_true, key=lambda i: diff[i])
                target_map[(objs_str, h)] = best
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    t_idx = np.array([target_map.get(k, -1) for k in keys], dtype=np.int64)
    valid = t_idx >= 0
    if not np.any(valid):
        return 0.0
    pts_series = sub["points"].iloc[np.flatnonzero(valid)]
    t_valid = t_idx[valid]
    unique_pts = pts_series.unique()
    pts_map = {s: json.loads(s) for s in unique_pts}
    vals = [pts_map[s][idx] for s, idx in zip(pts_series, t_valid)]
    return float(np.mean(vals))

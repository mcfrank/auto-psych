# name: heard_max_feature_competitor_points
# description: Mean points allocated to the most feature-rich true referent in heard-word scenes where multiple objects satisfy the heard word and differ in feature count, testing whether depth-2 RSA over-eliminates complex competitors via scalar implicature.
def test_statistic(df):
    heard_mask = df["heard"].notna()
    sub = df[heard_mask]
    if len(sub) == 0:
        return 0.0
    scenes = sub[["objects", "heard"]].drop_duplicates()
    target_idx_map = {}
    for objs_str, h_val in zip(scenes["objects"], scenes["heard"]):
        objs = json.loads(objs_str)
        h = int(h_val)
        true_indices = [i for i, o in enumerate(objs) if o[h] == 1]
        if len(true_indices) >= 2:
            fcounts = [sum(objs[i]) for i in true_indices]
            max_f = max(fcounts)
            min_f = min(fcounts)
            if max_f > min_f:
                max_indices = [i for i in true_indices if sum(objs[i]) == max_f]
                if len(max_indices) == 1:
                    target_idx_map[(objs_str, h)] = max_indices[0]
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    t_idx = np.array([target_idx_map.get(k, -1) for k in keys], dtype=np.int64)
    valid = t_idx >= 0
    if not np.any(valid):
        return 0.0
    pts_series = sub["points"].iloc[np.flatnonzero(valid)]
    t_valid = t_idx[valid]
    unique_pts = pts_series.unique()
    pts_map = {s: json.loads(s) for s in unique_pts}
    parsed = [pts_map[s] for s in pts_series]
    vals = [row[idx] for row, idx in zip(parsed, t_valid)]
    return float(np.mean(vals))

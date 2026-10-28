# name: prior_max_feature_singleton_points
# description: Mean difference in points on wordless prior trials (heard is NaN) between a unique maximal-feature singleton (possessing all scene features, feature_count == n_features) and the average intermediate-feature singleton (1 <= feature_count < n_features) in the same display.
def test_statistic(df):
    heard_int = df["heard"].fillna(-1).astype(int)
    scene_keys = (df["objects"] + "|" + heard_int.astype(str)).to_numpy()
    unique_keys = pd.unique(scene_keys)

    scene_weights = {}
    for uk in unique_keys:
        obj_str, h_str = uk.split("|")
        h = int(h_str)
        if h >= 0:
            continue
        objs = np.asarray(json.loads(obj_str), dtype=int)
        n_feat = objs.shape[1]
        fc = np.sum(objs, axis=1)
        dist_mat = np.sum(np.abs(objs[:, None, :] - objs[None, :, :]), axis=-1)
        is_sing = np.sum(dist_mat == 0, axis=1) == 1
        max_sing = is_sing & (fc == n_feat)
        mid_sing = is_sing & (fc >= 1) & (fc < n_feat)
        if np.any(max_sing) and np.any(mid_sing):
            w = max_sing.astype(np.float64) / float(np.sum(max_sing)) - mid_sing.astype(np.float64) / float(np.sum(mid_sing))
            scene_weights[uk] = w

    if not scene_weights:
        return 0.0

    pts_col = df["points"].to_numpy()
    total = 0.0
    count = 0
    for uk, w in scene_weights.items():
        idx = scene_keys == uk
        n_idx = int(np.sum(idx))
        if n_idx > 0:
            pts = np.asarray(json.loads("[" + ",".join(pts_col[idx]) + "]"), dtype=np.float64)
            total += float(np.sum(pts @ w))
            count += n_idx
    return float(total / count) if count > 0 else 0.0

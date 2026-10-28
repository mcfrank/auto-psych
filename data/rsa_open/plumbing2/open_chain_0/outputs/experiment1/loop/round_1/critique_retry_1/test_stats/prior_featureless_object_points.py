# name: prior_featureless_object_points
# description: Mean points allocated on wordless prior trials (heard is NaN) to completely featureless objects (feature_count == 0) in scenes that include a featureless object.
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
        fc = np.sum(objs, axis=1)
        bare = fc == 0
        if np.any(bare):
            scene_weights[uk] = bare.astype(np.float64)

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

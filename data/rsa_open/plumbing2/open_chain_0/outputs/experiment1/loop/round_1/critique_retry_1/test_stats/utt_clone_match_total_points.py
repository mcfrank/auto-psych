# name: utt_clone_match_total_points
# description: Mean total points allocated on utterance trials to identical duplicate (clone) literal matches (is_singleton == 0) in scenes where the heard word applies to both a clone pair and at least one singleton object (is_singleton == 1).
def test_statistic(df):
    heard_int = df["heard"].fillna(-1).astype(int)
    scene_keys = (df["objects"] + "|" + heard_int.astype(str)).to_numpy()
    unique_keys = pd.unique(scene_keys)

    scene_weights = {}
    for uk in unique_keys:
        obj_str, h_str = uk.split("|")
        h = int(h_str)
        if h < 0:
            continue
        objs = np.asarray(json.loads(obj_str), dtype=int)
        matches = objs[:, h] == 1
        dist_mat = np.sum(np.abs(objs[:, None, :] - objs[None, :, :]), axis=-1)
        dup_count = np.sum(dist_mat == 0, axis=1)
        clone_match = matches & (dup_count > 1)
        sing_match = matches & (dup_count == 1)
        if np.any(clone_match) and np.any(sing_match):
            scene_weights[uk] = clone_match.astype(np.float64)

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

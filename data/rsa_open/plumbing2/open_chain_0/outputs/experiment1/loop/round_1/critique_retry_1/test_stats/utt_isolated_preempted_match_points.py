# name: utt_isolated_preempted_match_points
# description: Mean points allocated on utterance trials to perceptually isolated literal matches (minimum Hamming distance >= 2 to all other display objects) that also possess a unique unmentioned feature (extension == 1), in scenes where another literal match has a close neighbor (minimum Hamming distance <= 1).
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
        n_obj, n_feat = objs.shape
        matches = objs[:, h] == 1
        if np.sum(matches) < 2:
            continue
        dist_mat = np.sum(np.abs(objs[:, None, :] - objs[None, :, :]), axis=-1)
        min_dist = np.min(dist_mat + np.eye(n_obj, dtype=int) * 999, axis=1)
        ext = np.sum(objs, axis=0)
        other_unique = np.zeros(n_obj, dtype=bool)
        for f in range(n_feat):
            if f != h and ext[f] == 1:
                other_unique |= (objs[:, f] == 1)
        target = matches & (min_dist >= 2) & other_unique
        comp = matches & (min_dist <= 1)
        if np.any(target) and np.any(comp):
            scene_weights[uk] = target.astype(np.float64)

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

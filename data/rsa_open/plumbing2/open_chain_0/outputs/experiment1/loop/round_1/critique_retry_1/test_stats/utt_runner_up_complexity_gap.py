# name: utt_runner_up_complexity_gap
# description: Mean difference in points per object between intermediate 2-feature literal matches (feature_count == 2) and high-complexity literal matches (feature_count >= 3) on utterance trials that also contain a 1-feature literal match (feature_count == 1).
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
        fc = np.sum(objs, axis=1)
        m1 = matches & (fc == 1)
        m2 = matches & (fc == 2)
        m3 = matches & (fc >= 3)
        if np.any(m1) and np.any(m2) and np.any(m3):
            w = m2.astype(np.float64) / float(np.sum(m2)) - m3.astype(np.float64) / float(np.sum(m3))
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

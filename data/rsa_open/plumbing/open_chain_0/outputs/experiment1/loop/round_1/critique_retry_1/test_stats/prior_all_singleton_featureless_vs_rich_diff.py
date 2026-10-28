# name: prior_all_singleton_featureless_vs_rich_diff
# description: Mean point difference between the least feature-rich object and the all-features object on prior trials (no word heard) where all four objects are unique singletons, testing whether the model's prior over-favors sparse/distinctive objects over feature-rich objects when singleton status is equal.
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
        if all(c == 1 for c in dup_counts) and len(objs) >= 2:
            fcounts = [sum(o) for o in objs]
            min_i = int(np.argmin(fcounts))
            max_i = int(np.argmax(fcounts))
            if min_i != max_i:
                scene_map[objs_str] = (min_i, max_i)
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
        min_i, max_i = matched[idx]
        vals.append(p[min_i] - p[max_i])
    return float(np.mean(vals))

# name: heard_false_referent_points_leakage
# description: Mean total points allocated to semantically false objects (objects lacking the heard feature) on trials where a word was heard and at least one object lacks that feature, testing whether the model's uniform lapse and Dirichlet-multinomial link mispredict points given to false referents.
def test_statistic(df):
    heard_mask = df["heard"].notna()
    sub = df[heard_mask]
    if len(sub) == 0:
        return 0.0
    scenes = sub[["objects", "heard"]].drop_duplicates()
    false_map = {}
    for objs_str, h_val in zip(scenes["objects"], scenes["heard"]):
        objs = json.loads(objs_str)
        h = int(h_val)
        false_indices = [i for i, o in enumerate(objs) if o[h] == 0]
        if len(false_indices) >= 1:
            false_map[(objs_str, h)] = false_indices
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    matched = [false_map.get(k, None) for k in keys]
    valid_idx = [i for i, m in enumerate(matched) if m is not None]
    if not valid_idx:
        return 0.0
    pts_series = sub["points"].iloc[valid_idx]
    unique_pts = pts_series.unique()
    pts_map = {s: json.loads(s) for s in unique_pts}
    vals = []
    for s, idx in zip(pts_series, valid_idx):
        p = pts_map[s]
        f_indices = matched[idx]
        vals.append(sum(p[i] for i in f_indices))
    return float(np.mean(vals))

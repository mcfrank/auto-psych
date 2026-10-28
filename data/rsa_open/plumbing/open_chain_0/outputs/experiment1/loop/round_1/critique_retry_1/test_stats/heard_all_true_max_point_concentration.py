# name: heard_all_true_max_point_concentration
# description: Mean maximum points placed on any single object on trials where the heard word is true of all four objects in the display, testing whether the model under- or over-predicts trial-level betting concentration when semantics rules out no referent.
def test_statistic(df):
    heard_mask = df["heard"].notna()
    sub = df[heard_mask]
    if len(sub) == 0:
        return 0.0
    scenes = sub[["objects", "heard"]].drop_duplicates()
    all_true_scenes = set()
    for objs_str, h_val in zip(scenes["objects"], scenes["heard"]):
        objs = json.loads(objs_str)
        h = int(h_val)
        if all(o[h] == 1 for o in objs):
            all_true_scenes.add((objs_str, h))
    keys = list(zip(sub["objects"], sub["heard"].astype(int)))
    valid_idx = [i for i, k in enumerate(keys) if k in all_true_scenes]
    if not valid_idx:
        return 0.0
    pts_series = sub["points"].iloc[valid_idx]
    unique_pts = pts_series.unique()
    pts_map = {s: max(json.loads(s)) for s in unique_pts}
    vals = [pts_map[s] for s in pts_series]
    return float(np.mean(vals))

# name: prior_multi_singleton_max_feature_probe_rating
# description: Mean second_rating on prior trials (heard is NaN) with at least two unique singletons where second_probe is the singleton with the highest feature count.
def test_statistic(df):
    prior_df = df[df["heard"].isna()]
    if prior_df.empty:
        return 0.0
    unique_objs = prior_df["objects"].unique()
    max_sing_map = {}
    for obj_str in unique_objs:
        objs = np.asarray(json.loads(obj_str), dtype=int)
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        if is_sing.sum() >= 2:
            sing_idx = np.where(is_sing)[0]
            fcnt = objs.sum(axis=1)
            max_f = fcnt[sing_idx].max()
            max_sing_map[obj_str] = int([i for i in sing_idx if fcnt[i] == max_f][0])
    if not max_sing_map:
        return 0.0
    max_target = prior_df["objects"].map(lambda s: max_sing_map.get(s, -1)).to_numpy()
    mask = prior_df["second_probe"].to_numpy() == max_target
    if not np.any(mask):
        return 0.0
    return float(prior_df["second_rating"].to_numpy()[mask].astype(float).mean())

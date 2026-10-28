# name: prior_sparse_vs_rich_singleton_choice_diff
# description: On prior trials (heard is NaN) with one unique singleton and N-1 identical clones, the rate of choosing the singleton when it has fewer features than the clones minus when it has more features than the clones.
def test_statistic(df):
    prior_df = df[df["heard"].isna()]
    if prior_df.empty:
        return 0.0
    unique_objs = prior_df["objects"].unique()
    scene_info = {}
    for obj_str in unique_objs:
        objs = np.asarray(json.loads(obj_str), dtype=int)
        n_obj = len(objs)
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        if is_sing.sum() == 1:
            sing_idx = int(np.where(is_sing)[0][0])
            fcnt = objs.sum(axis=1)
            clone_fcnt = fcnt[[i for i in range(n_obj) if i != sing_idx][0]]
            if fcnt[sing_idx] < clone_fcnt:
                scene_info[obj_str] = (1, sing_idx)
            elif fcnt[sing_idx] > clone_fcnt:
                scene_info[obj_str] = (-1, sing_idx)
    if not scene_info:
        return 0.0
    grp = prior_df["objects"].map(lambda s: scene_info.get(s, (0, -1))[0]).to_numpy()
    target = prior_df["objects"].map(lambda s: scene_info.get(s, (0, -1))[1]).to_numpy()
    chose_sing = (prior_df["choice"].to_numpy() == target).astype(float)
    sparse_mask = grp == 1
    rich_mask = grp == -1
    if not np.any(sparse_mask) or not np.any(rich_mask):
        return 0.0
    return float(chose_sing[sparse_mask].mean() - chose_sing[rich_mask].mean())

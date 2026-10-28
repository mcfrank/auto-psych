# name: prior_rich_minus_sparse_oddball_diff
# description: Difference in the proportion of choices selecting the unique singleton on no-word prior trials with one singleton and one clone group, comparing scenes where the singleton has more features than the clones minus scenes where the singleton has fewer features than the clones.
def test_statistic(df):
    obj_codes, obj_uniques = pd.factorize(df["objects"], sort=False)
    heard_vals = df["heard"].fillna(-1).to_numpy(dtype=np.int64)
    choices = df["choice"].to_numpy(dtype=np.int64)
    parsed_objs = [np.asarray(json.loads(s), dtype=np.int64) for s in obj_uniques]

    pair_keys = obj_codes * 32 + (heard_vals + 1)
    _, first_pos, inv = np.unique(pair_keys, return_index=True, return_inverse=True)

    n_scenes = len(first_pos)
    scene_group = np.zeros(n_scenes, dtype=np.int64)
    target_mask = np.zeros((n_scenes, 8), dtype=bool)

    for k, pos in enumerate(first_pos):
        h = int(heard_vals[pos])
        if h >= 0:
            continue
        arr = parsed_objs[obj_codes[pos]]
        n_obj = arr.shape[0]
        fc = np.sum(arr, axis=1)
        dup_count = np.sum(np.all(arr[:, None, :] == arr[None, :, :], axis=2), axis=1)
        sing_mask = dup_count == 1
        clone_mask = dup_count == (n_obj - 1)
        if np.sum(sing_mask) == 1 and np.sum(clone_mask) == (n_obj - 1):
            sing_fc = int(fc[sing_mask][0])
            clone_fc = int(fc[clone_mask][0])
            if sing_fc > clone_fc:
                scene_group[k] = 1
                target_mask[k, :n_obj] = sing_mask
            elif sing_fc < clone_fc:
                scene_group[k] = -1
                target_mask[k, :n_obj] = sing_mask

    row_group = scene_group[inv]
    rich_mask = row_group == 1
    sparse_mask = row_group == -1
    if not np.any(rich_mask) or not np.any(sparse_mask):
        return 0.0
    rich_rate = float(np.mean(target_mask[inv[rich_mask], choices[rich_mask]]))
    sparse_rate = float(np.mean(target_mask[inv[sparse_mask], choices[sparse_mask]]))
    return rich_rate - sparse_rate

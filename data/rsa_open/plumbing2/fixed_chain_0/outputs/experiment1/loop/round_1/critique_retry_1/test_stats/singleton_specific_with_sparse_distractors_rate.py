# name: singleton_specific_with_sparse_distractors_rate
# description: Proportion of choices selecting the minimal-feature singleton referent satisfying the heard word, in utterance scenes where all true referents are singletons of differing feature counts and false distractors are sparse (at most 1 feature each).
def test_statistic(df):
    obj_codes, obj_uniques = pd.factorize(df["objects"], sort=False)
    heard_vals = df["heard"].fillna(-1).to_numpy(dtype=np.int64)
    choices = df["choice"].to_numpy(dtype=np.int64)
    parsed_objs = [np.asarray(json.loads(s), dtype=np.int64) for s in obj_uniques]

    pair_keys = obj_codes * 32 + (heard_vals + 1)
    _, first_pos, inv = np.unique(pair_keys, return_index=True, return_inverse=True)

    n_scenes = len(first_pos)
    in_slice = np.zeros(n_scenes, dtype=bool)
    target_mask = np.zeros((n_scenes, 8), dtype=bool)

    for k, pos in enumerate(first_pos):
        h = int(heard_vals[pos])
        if h < 0:
            continue
        arr = parsed_objs[obj_codes[pos]]
        n_obj = arr.shape[0]
        true_mask = arr[:, h] == 1
        false_mask = ~true_mask
        if np.sum(true_mask) < 2 or np.sum(false_mask) < 1:
            continue
        fc = np.sum(arr, axis=1)
        dup_count = np.sum(np.all(arr[:, None, :] == arr[None, :, :], axis=2), axis=1)
        min_fc = np.min(fc[true_mask])
        max_fc = np.max(fc[true_mask])
        if (
            np.all(dup_count[true_mask] == 1)
            and max_fc > min_fc
            and np.all(fc[false_mask] <= 1)
        ):
            in_slice[k] = True
            target_mask[k, :n_obj] = true_mask & (fc == min_fc)

    row_mask = in_slice[inv]
    if not np.any(row_mask):
        return 0.0
    return float(np.mean(target_mask[inv[row_mask], choices[row_mask]]))

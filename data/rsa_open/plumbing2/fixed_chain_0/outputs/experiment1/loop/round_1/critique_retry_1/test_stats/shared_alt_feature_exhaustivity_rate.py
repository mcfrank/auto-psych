# name: shared_alt_feature_exhaustivity_rate
# description: Proportion of choices selecting a minimal-feature referent whose non-heard features are all shared across multiple objects, in utterance scenes where all objects satisfy the heard word, all have at least 2 features, and competing referents possess a unique non-heard feature.
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
        if not np.all(arr[:, h] == 1):
            continue
        fc = np.sum(arr, axis=1)
        if np.min(fc) < 2:
            continue
        ext = np.sum(arr, axis=0)
        has_unique = np.any((arr == 1) & (ext[None, :] == 1), axis=1)
        min_shared = (fc == np.min(fc)) & (~has_unique)
        if np.any(min_shared) and np.any(has_unique):
            in_slice[k] = True
            target_mask[k, :n_obj] = min_shared

    row_mask = in_slice[inv]
    if not np.any(row_mask):
        return 0.0
    return float(np.mean(target_mask[inv[row_mask], choices[row_mask]]))

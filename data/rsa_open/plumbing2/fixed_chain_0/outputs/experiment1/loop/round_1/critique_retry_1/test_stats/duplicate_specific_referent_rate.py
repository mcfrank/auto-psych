# name: duplicate_specific_referent_rate
# description: Proportion of choices selecting a minimal-feature (most specific) referent satisfying the heard word, in utterance scenes where the minimal-feature true referents are identical duplicates and a higher-feature true competitor is a singleton.
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
        if np.sum(true_mask) < 2:
            continue
        fc = np.sum(arr, axis=1)
        min_fc = np.min(fc[true_mask])
        max_fc = np.max(fc[true_mask])
        if max_fc <= min_fc:
            continue
        dup_count = np.sum(np.all(arr[:, None, :] == arr[None, :, :], axis=2), axis=1)
        min_true = true_mask & (fc == min_fc)
        high_true = true_mask & (fc > min_fc)
        if np.all(dup_count[min_true] > 1) and np.any(dup_count[high_true] == 1):
            in_slice[k] = True
            target_mask[k, :n_obj] = min_true

    row_mask = in_slice[inv]
    if not np.any(row_mask):
        return 0.0
    return float(np.mean(target_mask[inv[row_mask], choices[row_mask]]))

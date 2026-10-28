# name: two_feat_unique_alt_vs_three_feat_shared_rate
# description: Proportion of choices selecting the 2-feature referent satisfying the heard word, in utterance scenes where that 2-feature referent possesses a unique alternative feature (extension 1) while competing 3-feature true referents possess only shared features.
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
        fc = np.sum(arr, axis=1)
        ext = np.sum(arr, axis=0)
        has_unique = np.any((arr == 1) & (ext[None, :] == 1), axis=1)
        two_feat_uniq = true_mask & (fc == 2) & has_unique
        three_feat_shared = true_mask & (fc == 3) & (~has_unique)
        if (
            np.any(two_feat_uniq)
            and np.any(three_feat_shared)
            and np.all(two_feat_uniq | three_feat_shared | (~true_mask))
        ):
            in_slice[k] = True
            target_mask[k, :n_obj] = two_feat_uniq

    row_mask = in_slice[inv]
    if not np.any(row_mask):
        return 0.0
    return float(np.mean(target_mask[inv[row_mask], choices[row_mask]]))

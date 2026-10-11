# name: two_object_mumble_feature_rich_rate
# description: Rate of choosing the feature-richer object on mumble (prior) trials in two-object displays where objects have unequal feature counts.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        p_mask = (df["query"].to_numpy() == "prior")
        unique_objs = df.loc[p_mask, "objects"].unique()
        obj_info = {}
        for s in unique_objs:
            objs = json.loads(s)
            eligible = (len(objs) == 2) and (sum(objs[0]) != sum(objs[1]))
            target = (0 if sum(objs[0]) > sum(objs[1]) else 1) if eligible else None
            obj_info[s] = (eligible, target)

        p_indices = np.where(p_mask)[0]
        objs_arr = df["objects"].to_numpy()
        valid_indices = []
        target_indices = []
        for idx in p_indices:
            eligible, target = obj_info[objs_arr[idx]]
            if eligible:
                valid_indices.append(idx)
                target_indices.append(target)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(target_indices, dtype=int))

    valid_indices, target_indices = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    return float(np.mean(choices == target_indices))

# name: two_object_mumble_feature_rich_rate
# description: Rate of choosing the feature-richer object on mumble (prior) trials in two-object displays where objects have unequal feature counts.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        m_mask = (df["query"].to_numpy() == "prior")
        unique_disp = df.loc[m_mask, "objects"].drop_duplicates()
        disp_info = {}
        for s in unique_disp:
            objs = json.loads(s)
            eligible = (len(objs) == 2) and (sum(objs[0]) != sum(objs[1]))
            target = (0 if sum(objs[0]) > sum(objs[1]) else 1) if eligible else None
            disp_info[s] = (eligible, target)

        m_indices = np.where(m_mask)[0]
        objs_arr = df["objects"].to_numpy()
        valid_indices = []
        targets = []
        for idx in m_indices:
            eligible, target = disp_info[objs_arr[idx]]
            if eligible:
                valid_indices.append(idx)
                targets.append(target)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(targets, dtype=int))

    valid_indices, targets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    return float(np.mean(choices == targets))

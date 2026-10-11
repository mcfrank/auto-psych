# name: zero_feature_foil_selection_rate
# description: Rate of choosing a zero-feature object on utterance trials where the display contains an object with no features.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        u_mask = (df["query"].to_numpy() == "utterance")
        unique_objs = df.loc[u_mask, "objects"].unique()
        obj_info = {}
        for s in unique_objs:
            objs = json.loads(s)
            eligible = any(sum(o) == 0 for o in objs)
            is_zero = [sum(o) == 0 for o in objs] if eligible else None
            obj_info[s] = (eligible, is_zero)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()

        valid_indices = []
        is_zero_flags = []
        for idx in u_indices:
            s = objs_arr[idx]
            eligible, flags = obj_info[s]
            if eligible:
                valid_indices.append(idx)
                is_zero_flags.append(flags)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_zero_flags)

    valid_indices, is_zero_flags = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    is_zero = [flags[c] for flags, c in zip(is_zero_flags, choices)]
    return float(np.mean(is_zero))

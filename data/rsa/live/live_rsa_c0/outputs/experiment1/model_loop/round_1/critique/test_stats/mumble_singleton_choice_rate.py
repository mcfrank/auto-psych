# name: mumble_singleton_choice_rate
# description: Rate of choosing a unique singleton object on mumble (prior) trials where displays contain both singletons and duplicate objects.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        p_mask = (df["query"].to_numpy() == "prior")
        unique_objs = df.loc[p_mask, "objects"].unique()
        obj_info = {}
        for s in unique_objs:
            objs = json.loads(s)
            counts = [objs.count(x) for x in objs]
            eligible = any(c == 1 for c in counts) and any(c > 1 for c in counts)
            obj_info[s] = (eligible, [c == 1 for c in counts] if eligible else None)

        p_indices = np.where(p_mask)[0]
        objs_arr = df["objects"].to_numpy()
        valid_indices = []
        is_singleton_flags = []
        for idx in p_indices:
            s = objs_arr[idx]
            eligible, sing_flags = obj_info[s]
            if eligible:
                valid_indices.append(idx)
                is_singleton_flags.append(sing_flags)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_singleton_flags)

    valid_indices, is_singleton_flags = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chosen_singleton = [flags[c] for flags, c in zip(is_singleton_flags, choices)]
    return float(np.mean(chosen_singleton))

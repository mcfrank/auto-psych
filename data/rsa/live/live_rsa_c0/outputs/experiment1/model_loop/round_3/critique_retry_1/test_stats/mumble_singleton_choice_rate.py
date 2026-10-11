# name: mumble_singleton_choice_rate
# description: Rate of choosing a unique singleton object on mumble (prior) trials in displays containing both singleton objects and duplicate objects.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        p_mask = (df["query"].to_numpy() == "prior")
        unique_objs = df.loc[p_mask, "objects"].unique()
        obj_info = {}
        for s in unique_objs:
            objs = json.loads(s)
            n = len(objs)
            copies = [sum(objs[j] == objs[k] for k in range(n)) for j in range(n)]
            singletons = [j for j in range(n) if copies[j] == 1]
            duplicates = [j for j in range(n) if copies[j] > 1]
            eligible = (len(singletons) > 0) and (len(duplicates) > 0)
            obj_info[s] = (eligible, set(singletons))

        p_indices = np.where(p_mask)[0]
        objs_arr = df["objects"].to_numpy()
        valid_indices = []
        singleton_sets = []
        for idx in p_indices:
            eligible, singletons = obj_info[objs_arr[idx]]
            if eligible:
                valid_indices.append(idx)
                singleton_sets.append(singletons)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), singleton_sets)

    valid_indices, singleton_sets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    hits = [1.0 if c in s else 0.0 for c, s in zip(choices, singleton_sets)]
    return float(np.mean(hits))

# name: mumble_singleton_choice_rate
# description: Rate of choosing a unique singleton object on mumble (prior) trials in displays containing both singleton objects and duplicate objects.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        m_mask = (df["query"].to_numpy() == "prior")
        unique_disp = df.loc[m_mask, "objects"].drop_duplicates()
        disp_info = {}
        for s in unique_disp:
            objs = json.loads(s)
            n = len(objs)
            copies = [sum(objs[j] == objs[k] for k in range(n)) for j in range(n)]
            sing = [j for j in range(n) if copies[j] == 1]
            dup = [j for j in range(n) if copies[j] > 1]
            eligible = (len(sing) > 0) and (len(dup) > 0)
            disp_info[s] = (eligible, set(sing))

        m_indices = np.where(m_mask)[0]
        objs_arr = df["objects"].to_numpy()
        valid_indices = []
        sing_sets = []
        for idx in m_indices:
            eligible, s_set = disp_info[objs_arr[idx]]
            if eligible:
                valid_indices.append(idx)
                sing_sets.append(s_set)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), sing_sets)

    valid_indices, sing_sets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    hits = [1.0 if c in s else 0.0 for c, s in zip(choices, sing_sets)]
    return float(np.mean(hits))

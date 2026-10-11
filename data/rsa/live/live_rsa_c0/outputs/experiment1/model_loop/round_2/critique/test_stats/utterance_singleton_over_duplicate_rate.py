# name: utterance_singleton_over_duplicate_rate
# description: Rate of choosing a singleton matching object on utterance trials where the heard word is true of both a singleton object and at least one duplicate object.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        u_mask = (df["query"].to_numpy() == "utterance")
        unique_disp = df.loc[u_mask, ["objects", "utterance"]].drop_duplicates()
        disp_info = {}
        for s, u in unique_disp.itertuples(index=False):
            u = int(u)
            objs = json.loads(s)
            matching = [i for i, o in enumerate(objs) if o[u] == 1]
            counts = [objs.count(o) for o in objs]
            sing_m = [i for i in matching if counts[i] == 1]
            dup_m = [i for i in matching if counts[i] > 1]
            eligible = (len(sing_m) > 0) and (len(dup_m) > 0)
            if eligible:
                is_singleton_match = [counts[i] == 1 if i in matching else False for i in range(len(objs))]
                match_indices = set(matching)
            else:
                is_singleton_match = None
                match_indices = None
            disp_info[(s, u)] = (eligible, is_singleton_match, match_indices)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        is_singleton_flags = []
        match_sets = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, flags, m_set = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                is_singleton_flags.append(flags)
                match_sets.append(m_set)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_singleton_flags, match_sets)

    valid_indices, is_singleton_flags, match_sets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chosen_singleton = []
    for flags, m_set, c in zip(is_singleton_flags, match_sets, choices):
        if c in m_set:
            chosen_singleton.append(1.0 if flags[c] else 0.0)

    return float(np.mean(chosen_singleton)) if chosen_singleton else 0.0

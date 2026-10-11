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
            has_sing = any(counts[i] == 1 for i in matching)
            has_dup = any(counts[i] > 1 for i in matching)
            eligible = has_sing and has_dup
            disp_info[(s, u)] = (eligible, set(matching), counts)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        matching_sets = []
        counts_list = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, matching_set, counts = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                matching_sets.append(matching_set)
                counts_list.append(counts)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), matching_sets, counts_list)

    valid_indices, matching_sets, counts_list = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chose_singleton = []
    for c, matching_set, counts in zip(choices, matching_sets, counts_list):
        if c in matching_set:
            chose_singleton.append(counts[c] == 1)

    if not chose_singleton:
        return 0.0
    return float(np.mean(chose_singleton))

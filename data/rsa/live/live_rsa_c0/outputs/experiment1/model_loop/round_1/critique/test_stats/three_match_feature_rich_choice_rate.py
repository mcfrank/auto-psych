# name: three_match_feature_rich_choice_rate
# description: Rate of choosing the matching object with the highest feature count on utterance trials where the heard word is true of exactly three objects with varying feature counts.

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
            if len(matching) == 3:
                match_counts = [sum(objs[i]) for i in matching]
                eligible = (len(set(match_counts)) > 1)
                max_c = max(match_counts)
                is_max = {i: (sum(objs[i]) == max_c) for i in matching}
            else:
                eligible = False
                is_max = {}
            disp_info[(s, u)] = (eligible, is_max)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        is_max_maps = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, is_max_map = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                is_max_maps.append(is_max_map)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_max_maps)

    valid_indices, is_max_maps = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chose_richest = []
    for c, is_max_map in zip(choices, is_max_maps):
        if c in is_max_map:
            chose_richest.append(is_max_map[c])

    if not chose_richest:
        return 0.0
    return float(np.mean(chose_richest))

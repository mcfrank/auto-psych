# name: three_match_intermediate_feature_rate
# description: Rate of choosing the intermediate feature-count matching object on utterance trials where exactly three objects match the heard word with three strictly distinct feature counts.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        u_mask = (df["query"].to_numpy() == "utterance")
        unique_disp = df.loc[u_mask, ["objects", "utterance"]].drop_duplicates()
        disp_info = {}
        for s, u in unique_disp.itertuples(index=False):
            objs = json.loads(s)
            u = int(u)
            matching = [j for j, o in enumerate(objs) if o[u] == 1]
            eligible = (len(matching) == 3)
            target = None
            if eligible:
                fcounts = [sum(objs[j]) for j in matching]
                if len(set(fcounts)) == 3:
                    sorted_f = sorted(fcounts)
                    mid_f = sorted_f[1]
                    target = [j for j in matching if sum(objs[j]) == mid_f][0]
                else:
                    eligible = False
            disp_info[(s, u)] = (eligible, target, set(matching) if eligible else None)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)
        valid_indices = []
        targets = []
        match_sets = []
        for idx in u_indices:
            eligible, target, matching = disp_info[(objs_arr[idx], utt_arr[idx])]
            if eligible:
                valid_indices.append(idx)
                targets.append(target)
                match_sets.append(matching)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(targets, dtype=int), match_sets)

    valid_indices, targets, match_sets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    hits = [1.0 if c == t else 0.0 for c, t, m in zip(choices, targets, match_sets) if c in m]
    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))

# name: utterance_singleton_over_duplicate_rate
# description: Rate of choosing a unique singleton object on utterance trials where the heard word is true of both a singleton object and at least one duplicate object.

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
            n = len(objs)
            matching = [j for j in range(n) if objs[j][u] == 1]
            copies = [sum(objs[j] == objs[k] for k in range(n)) for j in range(n)]
            sing_matching = [j for j in matching if copies[j] == 1]
            dup_matching = [j for j in matching if copies[j] > 1]
            eligible = (len(sing_matching) > 0) and (len(dup_matching) > 0)
            disp_info[(s, u)] = (eligible, set(sing_matching), set(matching))

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)
        valid_indices = []
        sing_sets = []
        matching_sets = []
        for idx in u_indices:
            eligible, sing_matching, matching = disp_info[(objs_arr[idx], utt_arr[idx])]
            if eligible:
                valid_indices.append(idx)
                sing_sets.append(sing_matching)
                matching_sets.append(matching)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), sing_sets, matching_sets)

    valid_indices, sing_sets, matching_sets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    hits = [1.0 if c in s else 0.0 for c, s, m in zip(choices, sing_sets, matching_sets) if c in m]
    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))

# name: ubiquitous_utterance_max_feature_rate
# description: Rate of choosing the object with the maximum number of features on utterance trials where the heard word is true of every object in the display.

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
            eligible = all(o[u] == 1 for o in objs)
            if eligible:
                counts = [sum(o) for o in objs]
                max_c = max(counts)
                is_max = [c == max_c for c in counts]
            else:
                is_max = None
            disp_info[(s, u)] = (eligible, is_max)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        is_max_flags = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, flags = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                is_max_flags.append(flags)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_max_flags)

    valid_indices, is_max_flags = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chose_max = [flags[c] for flags, c in zip(is_max_flags, choices)]
    return float(np.mean(chose_max))

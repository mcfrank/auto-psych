# name: four_object_foil_error_rate
# description: Rate of choosing a non-matching foil object (an object lacking the heard feature) on utterance trials in four-object displays.

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
            eligible = (len(objs) == 4)
            disp_info[(s, u)] = (eligible, [o[u] == 0 for o in objs] if eligible else None)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        is_foil_flags = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, flags = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                is_foil_flags.append(flags)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_foil_flags)

    valid_indices, is_foil_flags = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chose_foil = [flags[c] for flags, c in zip(is_foil_flags, choices)]
    return float(np.mean(chose_foil))

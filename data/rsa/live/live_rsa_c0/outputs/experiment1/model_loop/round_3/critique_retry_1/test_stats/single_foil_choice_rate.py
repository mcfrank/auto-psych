# name: single_foil_choice_rate
# description: Rate of choosing the solitary non-matching foil object on utterance trials in displays with at least three objects where the heard word is true of all objects except exactly one.

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
            eligible = (len(matching) == len(objs) - 1) and (len(objs) >= 3)
            foil_idx = [j for j, o in enumerate(objs) if o[u] == 0][0] if eligible else None
            disp_info[(s, u)] = (eligible, foil_idx)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)
        valid_indices = []
        foil_indices = []
        for idx in u_indices:
            eligible, foil_idx = disp_info[(objs_arr[idx], utt_arr[idx])]
            if eligible:
                valid_indices.append(idx)
                foil_indices.append(foil_idx)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(foil_indices, dtype=int))

    valid_indices, foil_indices = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    return float(np.mean(choices == foil_indices))

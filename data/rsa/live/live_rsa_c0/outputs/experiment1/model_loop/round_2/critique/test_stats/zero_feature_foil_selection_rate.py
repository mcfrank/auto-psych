# name: zero_feature_foil_selection_rate
# description: Rate of choosing a zero-feature object on utterance trials where the display contains an object with no features.

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
            zero_indices = [i for i, o in enumerate(objs) if sum(o) == 0]
            eligible = (len(zero_indices) > 0)
            disp_info[(s, u)] = (eligible, [sum(o) == 0 for o in objs] if eligible else None)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        is_zero_flags = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, flags = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                is_zero_flags.append(flags)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), is_zero_flags)

    valid_indices, is_zero_flags = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    chose_zero = [flags[c] for flags, c in zip(is_zero_flags, choices)]
    return float(np.mean(chose_zero))

# name: popout_literal_accuracy
# description: Rate of choosing the unique matching object on popout utterance trials where the heard word is true of exactly one object in the display.

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
            matching = [j for j, x in enumerate(objs) if x[u] == 1]
            eligible = (len(matching) == 1)
            target = matching[0] if eligible else None
            disp_info[(s, u)] = (eligible, target)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)
        valid_indices = []
        targets = []
        for idx in u_indices:
            eligible, target = disp_info[(objs_arr[idx], utt_arr[idx])]
            if eligible:
                valid_indices.append(idx)
                targets.append(target)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(targets, dtype=int))

    valid_indices, targets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    return float(np.mean(choices == targets))

# name: two_object_scalar_implicature_rate
# description: Rate of choosing the sparser object (pragmatic scalar target) in two-object displays where the heard word is true of both objects with unequal feature counts.

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
            eligible = (len(objs) == 2) and (objs[0][u] == 1) and (objs[1][u] == 1) and (sum(objs[0]) != sum(objs[1]))
            if eligible:
                sparse_target = 0 if sum(objs[0]) < sum(objs[1]) else 1
            else:
                sparse_target = None
            disp_info[(s, u)] = (eligible, sparse_target)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)

        valid_indices = []
        target_indices = []
        for idx in u_indices:
            p = (objs_arr[idx], utt_arr[idx])
            eligible, target = disp_info[p]
            if eligible:
                valid_indices.append(idx)
                target_indices.append(target)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(target_indices, dtype=int))

    valid_indices, target_indices = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    return float(np.mean(choices == target_indices))

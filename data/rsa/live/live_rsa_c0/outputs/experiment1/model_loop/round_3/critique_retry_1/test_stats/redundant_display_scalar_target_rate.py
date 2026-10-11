# name: redundant_display_scalar_target_rate
# description: Rate of choosing the sparser matching object (scalar target) on utterance trials in displays with redundant feature dimensions where the heard word is true of multiple objects with unequal feature counts.

_CACHE = {}

def test_statistic(df):
    cache_key = (len(df), df["objects"].iloc[0], df["objects"].iloc[-1])
    if cache_key not in _CACHE:
        u_mask = (df["query"].to_numpy() == "utterance")
        unique_disp = df.loc[u_mask, ["objects", "utterance"]].drop_duplicates()
        disp_info = {}
        for s, u in unique_disp.itertuples(index=False):
            objs = json.loads(s)
            n_feats = len(objs[0])
            has_red = any(j != k and all(objs[o][j] == objs[o][k] for o in range(len(objs))) for j in range(n_feats) for k in range(j + 1, n_feats))
            u = int(u)
            matching = [j for j, o in enumerate(objs) if o[u] == 1]
            eligible = has_red and (len(matching) >= 2)
            target = None
            if eligible:
                fcounts = [sum(objs[j]) for j in matching]
                if min(fcounts) < max(fcounts):
                    min_f = min(fcounts)
                    sparser = [j for j in matching if sum(objs[j]) == min_f]
                    if len(sparser) == 1:
                        target = sparser[0]
                    else:
                        eligible = False
                else:
                    eligible = False
            disp_info[(s, u)] = (eligible, target, set(matching) if eligible else None)

        u_indices = np.where(u_mask)[0]
        objs_arr = df["objects"].to_numpy()
        utt_arr = df["utterance"].fillna(-1).to_numpy().astype(int)
        valid_indices = []
        targets = []
        matching_sets = []
        for idx in u_indices:
            eligible, target, matching = disp_info[(objs_arr[idx], utt_arr[idx])]
            if eligible:
                valid_indices.append(idx)
                targets.append(target)
                matching_sets.append(matching)

        _CACHE[cache_key] = (np.array(valid_indices, dtype=int), np.array(targets, dtype=int), matching_sets)

    valid_indices, targets, matching_sets = _CACHE[cache_key]
    if len(valid_indices) == 0:
        return 0.0

    choices = df["choice"].to_numpy()[valid_indices].astype(int)
    hits = [1.0 if c == t else 0.0 for c, t, m in zip(choices, targets, matching_sets) if c in m]
    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))

# name: all_true_utterance_rich_singleton_choice_rate
# description: On 4-object utterance trials where the heard word is true of every object and one unique singleton has strictly more features than the three identical clones, the rate of choosing the singleton.
def test_statistic(df):
    utt_df = df[df["heard"].notna()]
    if utt_df.empty:
        return 0.0
    pairs = utt_df[["objects", "heard"]].drop_duplicates()
    target_map = {}
    for obj_str, h_val in zip(pairs["objects"], pairs["heard"]):
        h = int(h_val)
        objs = np.asarray(json.loads(obj_str), dtype=int)
        if len(objs) != 4:
            continue
        truth = objs[:, h] == 1
        if not np.all(truth):
            continue
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        if is_sing.sum() == 1:
            sing_idx = int(np.where(is_sing)[0][0])
            fcnt = objs.sum(axis=1)
            clone_fcnt = fcnt[~is_sing][0]
            if fcnt[sing_idx] > clone_fcnt:
                target_map[(obj_str, h)] = sing_idx
    if not target_map:
        return 0.0
    keys = list(zip(utt_df["objects"], utt_df["heard"].astype(int)))
    choices = utt_df["choice"].to_numpy()
    vals = [float(c == target_map[k]) for k, c in zip(keys, choices) if k in target_map]
    if not vals:
        return 0.0
    return float(np.mean(vals))

# name: utterance_sparse_singleton_vs_rich_clones_clone_choice_rate
# description: On utterance trials where the heard word applies to one unique singleton and at least two identical clones with strictly more features than the singleton, the rate of choosing one of the true clones.
def test_statistic(df):
    utt_df = df[df["heard"].notna()]
    if utt_df.empty:
        return 0.0
    pairs = utt_df[["objects", "heard"]].drop_duplicates()
    clone_sets = {}
    for obj_str, h_val in zip(pairs["objects"], pairs["heard"]):
        h = int(h_val)
        objs = np.asarray(json.loads(obj_str), dtype=int)
        truth = objs[:, h] == 1
        diff = np.abs(objs[:, None, :] - objs[None, :, :]).sum(axis=2)
        is_sing = (diff == 0).sum(axis=1) == 1
        fcnt = objs.sum(axis=1)
        true_sing = np.where(truth & is_sing)[0]
        true_clones = np.where(truth & (~is_sing))[0]
        if len(true_sing) == 1 and len(true_clones) >= 2:
            if np.all(fcnt[true_clones] > fcnt[true_sing[0]]):
                clone_sets[(obj_str, h)] = set(int(i) for i in true_clones)
    if not clone_sets:
        return 0.0
    keys = list(zip(utt_df["objects"], utt_df["heard"].astype(int)))
    choices = utt_df["choice"].to_numpy()
    vals = [float(c in clone_sets[k]) for k, c in zip(keys, choices) if k in clone_sets]
    if not vals:
        return 0.0
    return float(np.mean(vals))

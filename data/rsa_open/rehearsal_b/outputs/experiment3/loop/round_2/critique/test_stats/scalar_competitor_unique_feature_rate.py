# name: scalar_competitor_unique_feature_rate
# description: Rate of choosing a 2-feature candidate possessing an unmentioned feature unique in the scene when competing against duplicate candidates; tests failure to account for contextual uniqueness of unmentioned features.
_CACHE = {}

def test_statistic(df):
    if "rows" not in _CACHE:
        unique = df[["objects", "heard"]].drop_duplicates()
        scene_targets = {}
        for _, r in unique.iterrows():
            if pd.notna(r["heard"]):
                objs = json.loads(r["objects"])
                h = int(r["heard"])
                compat = [i for i, o in enumerate(objs) if o[h] == 1]
                feat_counts = [sum(o) for o in objs]
                if len(compat) == len(objs):
                    n_feats = len(objs[0])
                    targets = set()
                    for i in compat:
                        if feat_counts[i] == 2:
                            for f in range(n_feats):
                                if f != h and objs[i][f] == 1:
                                    if sum(objs[j][f] for j in range(len(objs))) == 1:
                                        targets.add(i)
                    if len(targets) == 1:
                        scene_targets[(r["objects"], h)] = targets
        
        matching_rows = []
        target_matrix = []
        for idx, (o, h) in enumerate(zip(df["objects"], df["heard"])):
            h_val = -1 if pd.isna(h) else int(h)
            if (o, h_val) in scene_targets:
                matching_rows.append(idx)
                row_mask = [i in scene_targets[(o, h_val)] for i in range(10)]
                target_matrix.append(row_mask)
        _CACHE["rows"] = np.array(matching_rows, dtype=int)
        _CACHE["target_matrix"] = np.array(target_matrix, dtype=bool)
        _CACHE["arange"] = np.arange(len(matching_rows))
    
    rows = _CACHE["rows"]
    if len(rows) == 0:
        return 0.0
    choices = df["choice"].values[rows]
    hits = _CACHE["target_matrix"][_CACHE["arange"], choices]
    return float(np.mean(hits))

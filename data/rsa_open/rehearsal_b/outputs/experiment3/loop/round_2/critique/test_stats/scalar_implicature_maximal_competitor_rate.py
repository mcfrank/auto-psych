# name: scalar_implicature_maximal_competitor_rate
# description: Rate of choosing a 4-feature candidate on informative trials where exactly one minimal candidate and one 4-feature candidate are compatible and distractors are absent or duplicated; tests under-prediction of maximal referents in scalar implicature.
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
                dup_counts = [sum(1 for o in objs if o == obj) for obj in objs]
                feat_counts = [sum(o) for o in objs]
                if len(compat) == 2 and all(dup_counts[i] == 1 for i in compat):
                    fcs = [feat_counts[i] for i in compat]
                    if 4 in fcs:
                        foils = [i for i in range(len(objs)) if i not in compat]
                        if len(foils) == 0 or any(dup_counts[f] > 1 for f in foils):
                            max_i = [i for i in compat if feat_counts[i] == 4][0]
                            scene_targets[(r["objects"], h)] = {max_i}
        
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

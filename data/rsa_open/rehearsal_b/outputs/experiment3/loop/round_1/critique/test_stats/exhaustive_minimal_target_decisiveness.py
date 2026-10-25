# name: exhaustive_minimal_target_decisiveness
# description: Rate of choosing a unique 1-feature minimal target when competing against duplicated multi-feature competitors on informative trials; tests under-production of pragmatic contrast when a fully exhaustive description applies.
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
                min_compat = [i for i in compat if feat_counts[i] == 1]
                multi_compat = [i for i in compat if feat_counts[i] >= 3]
                if len(min_compat) == 1 and len(multi_compat) >= 2 and len(min_compat) + len(multi_compat) == len(compat):
                    if any(dup_counts[c] > 1 for c in multi_compat):
                        scene_targets[(r["objects"], h)] = {min_compat[0]}
        
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

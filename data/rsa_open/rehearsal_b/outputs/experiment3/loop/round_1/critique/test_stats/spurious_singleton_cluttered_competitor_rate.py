# name: spurious_singleton_cluttered_competitor_rate
# description: Rate of choosing a visual singleton candidate with redundant features when minimal compatible candidates are duplicated on informative trials; tests over-attraction to cluttered singletons from the singleton prior.
_CACHE = {}

def test_statistic(df):
    if "rows" not in _CACHE:
        unique = df[["objects", "heard"]].drop_duplicates()
        scene_targets = {}
        for _, r in unique.iterrows():
            if pd.notna(r["heard"]):
                objs = json.loads(r["objects"])
                h = int(r["heard"])
                dup_counts = [sum(1 for o in objs if o == obj) for obj in objs]
                feat_counts = [sum(o) for o in objs]
                compat = [i for i, o in enumerate(objs) if o[h] == 1]
                min_fc = min(feat_counts[i] for i in compat)
                has_dup_min = any(dup_counts[i] > 1 and feat_counts[i] == min_fc for i in compat)
                cluttered_singletons = [i for i in compat if dup_counts[i] == 1 and feat_counts[i] > min_fc]
                if has_dup_min and len(cluttered_singletons) == 1:
                    scene_targets[(r["objects"], h)] = {cluttered_singletons[0]}
        
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

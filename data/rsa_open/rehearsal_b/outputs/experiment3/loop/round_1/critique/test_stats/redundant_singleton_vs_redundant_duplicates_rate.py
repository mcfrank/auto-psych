# name: redundant_singleton_vs_redundant_duplicates_rate
# description: Rate of choosing a visual singleton candidate when competing against duplicated candidates where all candidates have redundant features (>1 feature) on informative trials; tests under-prediction of singleton salience in redundant scenes.
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
                all_fcs = [sum(o) for o in objs]
                singletons = [i for i in compat if dup_counts[i] == 1]
                dups = [i for i in compat if dup_counts[i] > 1]
                if len(singletons) == 1 and len(dups) >= 2:
                    s = singletons[0]
                    if all_fcs[s] > 1 and all(all_fcs[d] > 1 for d in dups):
                        scene_targets[(r["objects"], h)] = {s}
        
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

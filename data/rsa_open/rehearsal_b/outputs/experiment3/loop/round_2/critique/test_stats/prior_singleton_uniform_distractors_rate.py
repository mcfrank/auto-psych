# name: prior_singleton_uniform_distractors_rate
# description: Rate of choosing the visual singleton on uninformative prior trials in 6-object displays with 5 identical distractors; tests under-prediction of singleton salience in large uniform competitor sets.
_CACHE = {}

def test_statistic(df):
    if "rows" not in _CACHE:
        unique = df[["objects", "heard"]].drop_duplicates()
        scene_targets = {}
        for _, r in unique.iterrows():
            if pd.isna(r["heard"]):
                objs = json.loads(r["objects"])
                if len(objs) == 6:
                    dup_counts = [sum(1 for o in objs if o == obj) for obj in objs]
                    if dup_counts.count(1) == 1 and dup_counts.count(5) == 5:
                        s_idx = dup_counts.index(1)
                        scene_targets[(r["objects"], -1)] = {s_idx}
        
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

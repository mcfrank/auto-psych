# name: scalar_minimal_target_margin_contexts
# description: Mean points on minimal target [0, 0, 1] minus mean points on competitor objects across scalar implicature scenes with duplicate distractors when hearing feature 2.
def test_statistic(df):
    target_scenes = [
        ([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]], 2, 1, [2, 3]),
        ([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]], 2, 0, [1]),
        ([[0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1]], 2, 0, [3]),
    ]
    diffs = []
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if not pd.isna(sc["heard"]):
            objs = json.loads(sc["objects"])
            h = sc["heard"]
            for t_objs, t_h, t_idx, c_idx in target_scenes:
                if objs == t_objs and h == t_h:
                    sub = df[(df["objects"] == sc["objects"]) & (df["heard"] == sc["heard"])]
                    mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
                    t_val = mat[:, t_idx]
                    c_val = mat[:, c_idx].mean(axis=1)
                    diffs.append(t_val - c_val)
    if diffs:
        return float(np.mean(np.concatenate(diffs)))
    return 0.0

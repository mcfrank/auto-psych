# name: prior_dense_over_sparse_no_empty_scenes
# description: Mean points allocated to full 3-feature objects [1, 1, 1] minus mean points allocated to minimal 1-feature objects [0, 0, 1] on uninformative prior trials without empty objects.
def test_statistic(df):
    target_scenes = [
        ([[0, 0, 1], [0, 1, 1], [1, 1, 1]], 2, [0]),
        ([[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]], [2, 3], [0, 1]),
    ]
    diffs = []
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]):
            objs = json.loads(sc["objects"])
            for t_objs, dense_idx, sparse_idx in target_scenes:
                if objs == t_objs:
                    sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
                    mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
                    d_val = mat[:, dense_idx] if isinstance(dense_idx, int) else mat[:, dense_idx].mean(axis=1)
                    s_val = mat[:, sparse_idx].mean(axis=1)
                    diffs.append(d_val - s_val)
    if diffs:
        return float(np.mean(np.concatenate(diffs)))
    return 0.0

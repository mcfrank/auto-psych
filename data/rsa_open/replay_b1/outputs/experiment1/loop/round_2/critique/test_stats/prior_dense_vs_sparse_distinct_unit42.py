# name: prior_dense_vs_sparse_distinct_unit42
# description: Points allocated to dense 3-feature object [1, 1, 1] minus points allocated to sparse 1-feature object [0, 0, 1] on uninformative prior trials where all objects are distinct.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]) and json.loads(sc["objects"]) == [[0, 0, 1], [0, 1, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, 2] - mat[:, 0]
            return float(np.mean(diff))
    return 0.0

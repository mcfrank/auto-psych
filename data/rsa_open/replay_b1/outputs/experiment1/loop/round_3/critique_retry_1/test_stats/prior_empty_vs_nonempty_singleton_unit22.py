# name: prior_empty_vs_nonempty_singleton_unit22
# description: Points allocated to empty singleton [0, 0, 0] minus points allocated to 2-feature singleton [0, 1, 1] on uninformative prior trials with duplicate full distractors.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]) and json.loads(sc["objects"]) == [[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]]:
            sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            diff = mat[:, 0] - mat[:, 1]
            return float(np.mean(diff))
    return 0.0

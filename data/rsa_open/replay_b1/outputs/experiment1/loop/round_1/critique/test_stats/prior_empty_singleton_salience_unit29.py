# name: prior_empty_singleton_salience_unit29
# description: Mean points allocated to the empty singleton object [0, 0, 0] on prior trials with duplicate 1-feature objects and a 2-feature object.
def test_statistic(df):
    scenes = df[["objects", "heard"]].drop_duplicates()
    for _, sc in scenes.iterrows():
        if pd.isna(sc["heard"]) and json.loads(sc["objects"]) == [[0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 1, 0]]:
            sub = df[(df["objects"] == sc["objects"]) & df["heard"].isna()]
            mat = sub["points"].str.strip("[]").str.split(",", expand=True).to_numpy(dtype=float)
            return float(np.mean(mat[:, 0]))
    return 0.0

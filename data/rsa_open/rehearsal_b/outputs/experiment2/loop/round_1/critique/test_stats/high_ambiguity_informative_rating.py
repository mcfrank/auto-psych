# name: high_ambiguity_informative_rating
# description: Mean Likert rating across informative trials where the heard word is true of 4 or more objects; tests under-prediction of candidate likelihood under broad referential ambiguity.
_CACHE = {}

def test_statistic(df):
    if "idx" not in _CACHE:
        unique = df[["objects", "heard", "probe"]].drop_duplicates()
        matching = set()
        for _, r in unique.iterrows():
            if pd.notna(r["heard"]):
                objs = json.loads(r["objects"])
                h = int(r["heard"])
                if sum(o[h] for o in objs) >= 4:
                    matching.add((r["objects"], h, int(r["probe"])))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx"] = np.where([k in matching for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx"]].mean())

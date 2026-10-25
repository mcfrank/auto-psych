# name: prior_singleton_rating
# description: Mean Likert rating of a probed visual singleton on uninformative prior trials with no word heard; tests over-confidence of singleton prior in continuous likelihood ratings.
_CACHE = {}

def test_statistic(df):
    if "idx" not in _CACHE:
        unique = df[["objects", "heard", "probe"]].drop_duplicates()
        matching = set()
        for _, r in unique.iterrows():
            if pd.isna(r["heard"]):
                objs = json.loads(r["objects"])
                p = int(r["probe"])
                if sum(1 for o in objs if o == objs[p]) == 1:
                    other_objs = [o for i, o in enumerate(objs) if i != p]
                    if any(other_objs.count(o) > 1 for o in other_objs):
                        matching.add((r["objects"], -1, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx"] = np.where([k in matching for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx"]].mean())

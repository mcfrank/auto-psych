# name: universal_word_singleton_rating
# description: Mean Likert rating of a probed visual singleton when the heard word is true of all objects in the scene and distractors are duplicated; tests under-prediction from M-implicature penalty.
_CACHE = {}

def test_statistic(df):
    if "idx" not in _CACHE:
        unique = df[["objects", "heard", "probe"]].drop_duplicates()
        matching = set()
        for _, r in unique.iterrows():
            if pd.notna(r["heard"]):
                objs = json.loads(r["objects"])
                h = int(r["heard"])
                p = int(r["probe"])
                if all(o[h] == 1 for o in objs):
                    if sum(1 for o in objs if o == objs[p]) == 1:
                        distractors = [o for i, o in enumerate(objs) if i != p]
                        if len(set(tuple(o) for o in distractors)) == 1:
                            if sum(objs[p]) <= sum(distractors[0]):
                                matching.add((r["objects"], h, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx"] = np.where([k in matching for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx"]].mean())

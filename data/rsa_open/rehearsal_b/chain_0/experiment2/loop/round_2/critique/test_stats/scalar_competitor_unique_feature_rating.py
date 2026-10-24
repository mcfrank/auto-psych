# name: scalar_competitor_unique_feature_rating
# description: Mean Likert rating of a probed scalar competitor possessing an unmentioned unique feature when distractors sharing the heard word are duplicated; tests spurious singleton prior preservation.
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
                if objs[p][h] == 1 and sum(1 for o in objs if o == objs[p]) == 1:
                    comp_objs = [o for o in objs if o[h] == 1]
                    min_fc = min(sum(o) for o in comp_objs)
                    if sum(objs[p]) == min_fc + 1:
                        dup_min = any(comp_objs.count(o) > 1 for o in comp_objs if sum(o) == min_fc)
                        has_unique = any(objs[p][f] == 1 and sum(o[f] for o in objs) == 1 for f in range(len(objs[0])) if f != h)
                        if dup_min and has_unique:
                            matching.add((r["objects"], h, p))
        keys = [(o, -1 if pd.isna(h) else int(h), p) for o, h, p in zip(df["objects"], df["heard"], df["probe"])]
        _CACHE["idx"] = np.where([k in matching for k in keys])[0]
    return float(df["rating"].iloc[_CACHE["idx"]].mean())

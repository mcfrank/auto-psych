# name: target_is_example_yes_rate
# description: Yes rate among trials whose target is literally one of the shown examples; t_observed > null_mean means the model under-produces "yes" for repeated examples (it lacks an identity/membership term).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).astype(str)
    uniq = sets.unique()
    m = {s: set(s.split(",")) for s in uniq}
    keys = sets + "|" + t
    pairs = pd.Series(keys.unique())
    mem = {k: (k.split("|")[1] in m[k.split("|")[0]]) for k in pairs}
    mask = keys.map(mem).astype(bool).values
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())

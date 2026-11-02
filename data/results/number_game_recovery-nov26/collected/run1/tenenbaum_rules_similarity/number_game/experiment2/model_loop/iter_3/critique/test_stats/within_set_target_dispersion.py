# name: within_set_target_dispersion
# description: Mean over sets (>=3 distinct targets) of the variance across targets of the per-(set,target) yes rate; observed > null means people's generalisation within a set is more sharply all-or-none across targets than the model's, observed < null means flatter.
def test_statistic(df):
    c = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    v = c.groupby("number_set")["rating"].agg(["var", "size"])
    v = v[v["size"] >= 3]
    return float(v["var"].mean())

# name: yes_rate_target_in_examples
# description: Yes rate among trials whose target is itself one of the example numbers; observed below the null means the model is too confident that shown examples belong to the concept (humans say no more often).
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    tgt = df["target"].astype(int).values
    mask = (sets == tgt[:, None]).any(axis=1)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())

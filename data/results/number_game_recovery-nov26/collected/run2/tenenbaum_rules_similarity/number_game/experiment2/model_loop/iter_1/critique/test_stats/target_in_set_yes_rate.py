# name: target_in_set_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed < null means the model overpredicts endorsement of shown examples (people say no to them more than posterior membership implies, e.g. a no-repeat belief or lapses), observed > null means the model's noise share makes it underpredict.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    hit = (ex == t[:, None]).any(axis=1)
    r = df["rating"].to_numpy().astype(float)
    if hit.sum() == 0:
        return 0.0
    return float(r[hit].mean())

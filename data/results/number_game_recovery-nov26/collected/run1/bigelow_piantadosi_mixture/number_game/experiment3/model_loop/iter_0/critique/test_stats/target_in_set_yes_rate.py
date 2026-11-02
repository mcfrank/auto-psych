# name: target_in_set_yes_rate
# description: Yes rate for targets that equal one of the shown examples; observed < model means the model over-endorses the examples themselves (people do not always say yes to a shown number), observed > model the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tg = df["target"].astype(float).to_numpy()
    inset = (ex == tg[:, None]).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    return float(r[inset].mean()) if inset.any() else 0.0

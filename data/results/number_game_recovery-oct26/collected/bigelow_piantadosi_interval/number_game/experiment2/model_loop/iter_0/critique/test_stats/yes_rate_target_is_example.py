# name: yes_rate_target_is_example
# description: Yes rate on trials where the target is itself one of the shown examples; observed > null means people endorse already-seen examples more reliably than the model's noisy-example/tempered posterior predicts (observed < null means less).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    tg = df["target"].astype(int).values
    hit = (ex == tg[:, None]).any(axis=1)
    if hit.sum() == 0:
        return 0.0
    return float(df["rating"].values[hit].mean())

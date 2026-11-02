# name: target_in_set_yes_rate
# description: Yes rate on trials whose target is itself one of the shown examples; observed below the model's null mean means the model over-predicts "yes" for already-seen examples (humans say no to them far more than a near-certain concept-membership model allows).
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    m = (ex == t[:, None]).any(axis=1)
    return float(r[m].mean()) if m.any() else 0.0

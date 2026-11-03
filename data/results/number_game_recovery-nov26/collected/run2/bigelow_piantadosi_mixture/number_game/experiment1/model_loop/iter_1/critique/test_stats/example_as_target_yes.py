# name: example_as_target_yes
# description: Yes rate on trials whose target is itself one of the shown examples; observed below null means the model over-predicts certain yes for already-seen examples (people do not treat examples as guaranteed members).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    t = df["target"].astype(int).to_numpy()[:, None]
    hit = (ex.to_numpy() == t).any(axis=1)
    if hit.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[hit].mean())

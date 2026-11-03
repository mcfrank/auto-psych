# name: outside_range_yes_rate
# description: Yes-rate among targets lying outside [min, max] of the example set; observed > model means people extend generalisation beyond the examples' range more than the model's intervals/rules allow (observed < model: people are more range-bound).
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    tgt = df["target"].astype(float).to_numpy()
    out = (tgt < np.nanmin(sets, axis=1)) | (tgt > np.nanmax(sets, axis=1))
    r = df["rating"].to_numpy().astype(float)
    if out.sum() == 0:
        return 0.0
    return float(r[out].mean())

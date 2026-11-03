# name: near_target_yes_rate
# description: Yes rate for targets within 1-5 of the nearest example (not equal to an example); observed > null means the model under-generalizes by numeric proximity, observed < null means it over-generalizes to neighbours.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    dist = np.where(~np.isnan(s), np.abs(s - t), np.inf).min(axis=1)
    m = (dist >= 1) & (dist <= 5)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())

# name: single_example_near_yes
# description: For one-example sets, yes-rate on targets within 30 of the example (excluding the example) minus yes-rate on targets more than 40 away; observed > null means people show a steeper similarity gradient around a lone example than the model, observed < null means a flatter one.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",")
    x = pd.to_numeric(s.where(one, "nan"), errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    d = np.abs(t - x)
    a = one.to_numpy() & (d > 0) & (d <= 30); b = one.to_numpy() & (d > 40)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())

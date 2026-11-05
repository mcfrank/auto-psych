# name: single_example_multiples_rate
# description: Yes rate on one-example sets (example x>=2) for non-example targets that are multiples of x; observed above null means the model still under-generalizes a lone example to its arithmetic rule (6->18, 25->100).
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    sub = df[n == 1]
    x = pd.to_numeric(sub["number_set"], errors="coerce").to_numpy()
    t = sub["target"].astype(int).to_numpy()
    r = sub["rating"].to_numpy().astype(float)
    m = (x >= 2) & (t != x) & (t % np.maximum(x, 1) == 0)
    return float(r[m].mean()) if m.sum() > 0 else 0.0

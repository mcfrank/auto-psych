# name: single_example_proximity_gradient
# description: For single-example sets, yes-rate on non-example targets within 10 of the example minus yes-rate on targets more than 30 away; observed > model means people generalize from one example by numeric magnitude similarity more than the model's rule/interval posterior predicts, observed < model means magnitude matters less than predicted.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").values
    x = pd.to_numeric(s.where(single, "nan"), errors="coerce").values
    tgt = df["target"].astype(float).values
    d = np.abs(tgt - x)
    near = single & (d > 0) & (d <= 10)
    far = single & (d > 30)
    r = df["rating"].astype(float).values
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())

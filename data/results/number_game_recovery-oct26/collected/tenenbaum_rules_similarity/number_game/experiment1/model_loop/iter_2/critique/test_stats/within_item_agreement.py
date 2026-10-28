# name: within_item_agreement
# description: Mean, over (set, target) items, of the pairwise agreement rate among the responses to that item; observed above null_mean means people are more consistent per item than the model (its per-item probabilities are too moderate, close to 0.5), below means people disagree more than the model allows.
def test_statistic(df):
    g = df.groupby([df["number_set"].astype(str), df["target"].astype(int)])["rating"].agg(["sum", "count"])
    k = g["sum"].to_numpy(float); n = g["count"].to_numpy(float)
    m = n >= 2
    k, n = k[m], n[m]
    agree = (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))
    return float(np.mean(agree)) if len(agree) else 0.0

# name: four_example_far_target_yes_rate
# description: For 4-example sets, yes-rate on targets more than 20 outside [min,max] of the examples (excluding targets <=10 and target 100); observed > null means the model under-produces far generalisation once four examples establish a rule (people extend well-supported rules across magnitudes more than the model's interval mass allows).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    sets = df["number_set"].astype(str).str.split(",").apply(lambda l: [int(x) for x in l])
    k = sets.str.len().to_numpy()
    mn = sets.apply(min).to_numpy(); mx = sets.apply(max).to_numpy()
    dist = np.maximum(0, np.maximum(mn - t, t - mx))
    m = (k == 4) & (dist > 20) & (t > 10) & (t < 100)
    r = df["rating"].to_numpy()
    return float(r[m].mean()) if m.any() else 0.0

# name: single_example_mid_distance_yes
# description: Yes-rate among one-example sets for targets 15-50 away from the example; observed < null means people generalise less to moderately distant numbers from a lone example than the model's range/rule guesses imply, observed > null means more.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).values
    t = df["target"].astype(float).values
    d = np.abs(t - x)
    m = single.values & (d >= 15) & (d <= 50)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())

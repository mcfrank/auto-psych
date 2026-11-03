# name: first_encounter_of_set_yes
# description: Yes rate on the trial where a participant first sees a given number_set minus yes rate on their later trials with that set; observed above null means first judgments about a set are more liberal (later ones stricter) than the model's order-free responding predicts, below null the reverse.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(ti == 0)
    key = pd.Series(pid.astype(str)) + "|" + df["number_set"].astype(str).reset_index(drop=True)
    first = ~key.duplicated().to_numpy()
    multi = key.map(key.value_counts()).to_numpy() > 1
    r = df["rating"].to_numpy().astype(float)
    a, b = first & multi, (~first) & multi
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())

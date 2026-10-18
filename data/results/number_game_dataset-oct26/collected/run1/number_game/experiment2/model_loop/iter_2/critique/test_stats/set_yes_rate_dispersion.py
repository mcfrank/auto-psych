# name: set_yes_rate_dispersion
# description: SD across number_sets of the set-level yes rate (targets that are not examples); t_observed above null_mean means people's overall generalization breadth differs more between sets (some sets almost all-no, others almost all-yes) than the model's features produce.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int)
    padded = "," + sets.str.replace(" ", "") + ","
    in_set = np.array([p.find("," + str(v) + ",") >= 0 for p, v in zip(padded.values, t.values)])
    d = df.loc[~in_set]
    rates = d.groupby("number_set")["rating"].mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.std())

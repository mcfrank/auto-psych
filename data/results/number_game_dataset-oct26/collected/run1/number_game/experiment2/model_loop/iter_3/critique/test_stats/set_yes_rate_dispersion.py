# name: set_yes_rate_dispersion
# description: SD across number_sets of the set-level yes rate (targets that are not examples); t_observed above null_mean means generalization breadth differs more between sets (some nearly all-no, others nearly all-yes) than the model's features produce; below means the model over-differentiates sets.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    in_set = np.array([v in [int(x) for x in s.split(",")] for s, v in zip(sets, t)])
    d = df.loc[~in_set]
    rates = d.groupby("number_set")["rating"].mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.std())

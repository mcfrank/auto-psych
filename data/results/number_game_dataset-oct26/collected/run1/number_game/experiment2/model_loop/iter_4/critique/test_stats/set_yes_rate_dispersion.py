# name: set_yes_rate_dispersion
# description: SD across number_sets of the set-level yes rate on targets that are not examples; t_observed above null_mean means generalization breadth still differs more between sets (some near all-no, others near all-yes) than the rule-gated model produces; below means the model over-differentiates sets.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy()
    in_set = (ex == t[:, None]).any(axis=1)
    sub = df.loc[~in_set]
    rates = sub.groupby("number_set")["rating"].mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.std())

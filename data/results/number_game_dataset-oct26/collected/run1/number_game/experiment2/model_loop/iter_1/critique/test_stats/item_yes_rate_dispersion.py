# name: item_yes_rate_dispersion
# description: Standard deviation across (set, target) items of the item-level yes rate; t_observed > null_mean means human responses are more all-or-none / extreme across items than the model's graded predictions (model under-produces item polarization).
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean()
    return float(g.std(ddof=0))

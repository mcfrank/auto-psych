# name: between_set_yes_rate_sd
# description: Standard deviation across number sets of each set's mean yes rate; observed > model means people treat some sets as far more (or less) generous than the model, i.e. the model under-produces set-level extremity (near-all-yes / near-all-no sets).
def test_statistic(df):
    m = df.groupby("number_set")["rating"].mean()
    return float(m.std(ddof=0))

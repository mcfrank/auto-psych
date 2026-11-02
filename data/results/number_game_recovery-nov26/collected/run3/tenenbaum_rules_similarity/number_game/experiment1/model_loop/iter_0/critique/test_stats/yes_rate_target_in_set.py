# name: yes_rate_target_in_set
# description: Proportion of yes answers when the target is one of the shown examples; observed below the model's mean means the model over-predicts certainty (no lapse/noise) for example members.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        E[i, :len(v)] = v
    X = E[codes]  # (rows, 4) examples, NaN padded
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(X), axis=1)
    m = np.any(X == t[:, None], axis=1)
    return float(r[m].mean()) if m.any() else 0.0

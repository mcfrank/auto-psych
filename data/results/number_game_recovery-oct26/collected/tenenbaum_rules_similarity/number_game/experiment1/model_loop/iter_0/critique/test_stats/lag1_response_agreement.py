# name: lag1_response_agreement
# description: Proportion of consecutive rows (same session: trial_index increases by 1) whose yes/no responses agree, minus the agreement expected from the two rows' marginal yes-rate; the model treats responses as independent, so observed above the null means humans perseverate on their previous answer (below: alternate).
def test_statistic(df):
    ti = df["trial_index"].values.astype(float)
    r = df["rating"].values.astype(float)
    if len(r) < 3:
        return 0.0
    cons = np.diff(ti) == 1
    if cons.sum() == 0:
        return 0.0
    a = r[:-1][cons]
    b = r[1:][cons]
    agree = np.mean(a == b)
    p = r.mean()
    return float(agree - (p * p + (1 - p) * (1 - p)))

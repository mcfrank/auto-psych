# name: majority_digit_rule_with_outlier_yes
# description: In 4-example sets where exactly 3 examples share a final digit and one does not, yes rate for non-example targets with that majority final digit; observed below null means the outlier undermines the rule more for people than the model's reliability belief predicts, above means people discount the outlier more.
def test_statistic(df):
    s = df["number_set"].astype(str)
    u = pd.unique(s)
    codes = pd.Categorical(s, categories=u).codes
    U = np.full((len(u), 4), np.nan)
    for i, x in enumerate(u):
        v = [int(float(z)) for z in x.split(",") if z.strip() != ""][:4]
        U[i, :len(v)] = v
    E = U[codes]
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    Ed = np.mod(E, 10)
    td = np.mod(t, 10)
    shares_last = np.any(Ed == td[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    first = Ed[:, 0]
    all_same_last = np.all((Ed == first[:, None]) | np.isnan(Ed), axis=1)
    cnt = np.stack([np.sum(Ed == d, axis=1) for d in range(10)], axis=1)
    maj = np.argmax(cnt, axis=1)
    m = (n == 4) & (cnt.max(axis=1) == 3) & (td == maj) & ~in_set
    return float(y[m].mean()) if m.any() else 0.0

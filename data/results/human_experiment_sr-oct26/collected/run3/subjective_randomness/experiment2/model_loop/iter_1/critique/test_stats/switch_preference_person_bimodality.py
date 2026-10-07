# name: switch_preference_person_bimodality
# description: Across participants, the fraction whose rate of choosing the more-switching sequence (unequal-switch trials) is below 0.4 (streak-preferers); observed > null means the model's normal population of switch beliefs under-produces a distinct subgroup preferring fewer switches.
def _sw(s):
    return sum(1 for x, y in zip(s, s[1:]) if x != y)
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    u = pd.unique(pd.concat([a, b])); sw = {s: _sw(s) for s in u}
    ka = a.map(sw).to_numpy(); kb = b.map(sw).to_numpy()
    y = df["chose_left"].to_numpy()
    d = ka != kb
    more = pd.Series(np.where(ka > kb, y, 1 - y)[d].astype(float))
    r = more.groupby(df["participant_id"].to_numpy()[d]).mean()
    return float((r < 0.4).mean())

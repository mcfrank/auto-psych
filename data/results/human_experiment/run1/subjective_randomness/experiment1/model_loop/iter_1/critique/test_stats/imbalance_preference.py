# name: imbalance_preference
# description: Choice rate of sequence A when it has an intermediate imbalance (0.5) and B is more balanced, to check if the model's quadratic penalty on p captures human preferences.
def test_statistic(df):
    mask = (df['imbalance_a'] == 0.5) & (df['imbalance_b'] < 0.5)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5

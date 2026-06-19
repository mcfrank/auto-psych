# name: head_imbalance_b_choice_rate
# description: Mean chose_left when imbalance_b > 0.3 and imbalance_a < 0.1 (B is head-biased, A is balanced); tests calibration of the biased-coin component by isolating trials where B's head proportion clearly departs from 0.5.
def test_statistic(df):
    mask = (df['imbalance_b'] > 0.3) & (df['imbalance_a'] < 0.1)
    sub = df[mask]
    if len(sub) == 0:
        return float('nan')
    return float(sub['chose_left'].mean())

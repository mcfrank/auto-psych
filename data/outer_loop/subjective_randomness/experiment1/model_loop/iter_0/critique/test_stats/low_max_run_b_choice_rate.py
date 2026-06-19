# name: low_max_run_b_choice_rate
# description: Mean chose_left when max_run_norm_b < 0.2 (B has only very short runs, appears locally balanced); tests if humans judge B as more random when it lacks salient streaks.
def test_statistic(df):
    mask = df['max_run_norm_b'] < 0.2
    sub = df[mask]
    if len(sub) == 0:
        return float('nan')
    return float(sub['chose_left'].mean())

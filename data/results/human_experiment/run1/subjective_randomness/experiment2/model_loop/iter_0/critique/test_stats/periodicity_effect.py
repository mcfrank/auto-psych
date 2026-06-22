# name: periodicity_effect
# description: Covariance between the difference in periodicity and the choice of the left sequence.
def test_statistic(df):
    diff_periodicity = df["periodicity_a"] - df["periodicity_b"]
    if len(df) <= 1:
        return 0.0
    cov = df["chose_left"].cov(diff_periodicity)
    import pandas as pd

    return float(cov) if pd.notna(cov) else 0.0

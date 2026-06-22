import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
for col in ['periodicity', 'p_alts', 'max_run_norm', 'imbalance']:
    df[f'diff_{col}'] = df[f'{col}_a'] - df[f'{col}_b']

X = df[['diff_periodicity', 'diff_p_alts', 'diff_max_run_norm', 'diff_imbalance']]
y = df['chose_left']

model = LogisticRegression().fit(X, y)
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"{name}: {coef:.3f}")

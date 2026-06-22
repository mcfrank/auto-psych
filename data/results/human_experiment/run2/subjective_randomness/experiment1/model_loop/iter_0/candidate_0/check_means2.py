import pandas as pd

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
print("Total rows:", len(df))
print("When imbalance_a > imbalance_b:")
sub = df[df['imbalance_a'] > df['imbalance_b']]
print(f"Count: {len(sub)}")
print(f"Mean chose_left: {sub['chose_left'].mean()}")

sub2 = df[df['imbalance_a'] < df['imbalance_b']]
print("When imbalance_a < imbalance_b:")
print(f"Count: {len(sub2)}")
print(f"Mean chose_left: {sub2['chose_left'].mean()}")


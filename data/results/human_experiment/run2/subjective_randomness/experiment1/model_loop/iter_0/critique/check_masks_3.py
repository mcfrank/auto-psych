import pandas as pd
import numpy as np
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def check_mask(mask_name, mask):
    if mask.sum() > 0:
        print(f"{mask_name}: {mask.sum()} trials, mean chose_left = {df.loc[mask, 'chose_left'].mean():.3f}")
    else:
        print(f"{mask_name}: 0 trials")

# over_alt vs under_alt difference
print("corr over_alt vs under_alt", np.corrcoef((df['p_alts_a'] - 0.6) - (df['p_alts_b'] - 0.6), df['chose_left'])[0, 1])

# effect of n difference
print("corr n_a - n_b", np.corrcoef(df['n_a'] - df['n_b'], df['chose_left'])[0, 1])

mask_imb_n = (df['imbalance_a'] == df['imbalance_b']) & (df['n_a'] > df['n_b'])
check_mask("same imbalance, A is longer", mask_imb_n)

# Let's write the test statistics and evaluate them on data

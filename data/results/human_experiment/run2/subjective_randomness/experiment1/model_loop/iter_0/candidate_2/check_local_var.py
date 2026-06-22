import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def local_h_var(seq, w=3):
    if len(seq) < w: return 0.0
    counts = []
    for i in range(len(seq) - w + 1):
        counts.append(seq[i:i+w].count('H'))
    return np.var(counts)

def local_alt_var(seq, w=3):
    if len(seq) < w: return 0.0
    alts = []
    for i in range(len(seq) - w + 1):
        sub = seq[i:i+w]
        a = sum(1 for j in range(len(sub)-1) if sub[j] != sub[j+1])
        alts.append(a)
    return np.var(alts)

df['h_var_3_a'] = df['sequence_a'].apply(lambda x: local_h_var(x, 3))
df['h_var_3_b'] = df['sequence_b'].apply(lambda x: local_h_var(x, 3))
df['alt_var_4_a'] = df['sequence_a'].apply(lambda x: local_alt_var(x, 4))
df['alt_var_4_b'] = df['sequence_b'].apply(lambda x: local_alt_var(x, 4))

print("Corr(h_var_3_a - h_var_3_b, chose_left):", np.corrcoef(df['h_var_3_a'] - df['h_var_3_b'], df['chose_left'])[0, 1])
print("Corr(alt_var_4_a - alt_var_4_b, chose_left):", np.corrcoef(df['alt_var_4_a'] - df['alt_var_4_b'], df['chose_left'])[0, 1])

# What about variance of run lengths?
def run_len_var(seq):
    runs = []
    curr = seq[0]
    l = 1
    for c in seq[1:]:
        if c == curr:
            l += 1
        else:
            runs.append(l)
            curr = c
            l = 1
    runs.append(l)
    return np.var(runs) if len(runs) > 1 else 0.0

df['run_var_a'] = df['sequence_a'].apply(run_len_var)
df['run_var_b'] = df['sequence_b'].apply(run_len_var)
print("Corr(run_var_a - run_var_b, chose_left):", np.corrcoef(df['run_var_a'] - df['run_var_b'], df['chose_left'])[0, 1])


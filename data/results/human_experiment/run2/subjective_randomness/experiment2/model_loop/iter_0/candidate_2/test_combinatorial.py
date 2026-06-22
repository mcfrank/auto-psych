import math
import pandas as pd

def log_multinomial(s):
    runs = []
    current_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    counts = {1:0, 2:0, 3:0}
    for r in runs:
        k = r if r <= 2 else 3
        counts[k] += 1
    
    R = sum(counts.values())
    val = math.lgamma(R + 1) - math.lgamma(counts[1] + 1) - math.lgamma(counts[2] + 1) - math.lgamma(counts[3] + 1)
    return val

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/responses.csv')
for i, row in df.head(10).iterrows():
    sa, sb = row['sequence_a'], row['sequence_b']
    ca = log_multinomial(sa)
    cb = log_multinomial(sb)
    print(f"{sa} (n={row['n_a']}): score={ca:.2f}")
    print(f"{sb} (n={row['n_b']}): score={cb:.2f}")
    print(f"Chose left: {row['chose_left']} (Score diff: {ca - cb:.2f})")
    print()

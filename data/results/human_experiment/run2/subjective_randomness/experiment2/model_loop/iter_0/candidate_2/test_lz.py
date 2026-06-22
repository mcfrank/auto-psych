import pandas as pd
import numpy as np

def lz78(s):
    if not s: return 0
    chunks = set()
    i = 0
    count = 0
    while i < len(s):
        j = 1
        while i + j <= len(s) and s[i:i+j] in chunks:
            j += 1
        chunks.add(s[i:i+j])
        i += j
        count += 1
    return count

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/responses.csv')
for i, row in df.head(10).iterrows():
    sa, sb = row['sequence_a'], row['sequence_b']
    print(f"{sa} (n={row['n_a']}): LZ={lz78(sa)}, Imb={row['imbalance_a']:.2f}")
    print(f"{sb} (n={row['n_b']}): LZ={lz78(sb)}, Imb={row['imbalance_b']:.2f}")
    print(f"Chose left: {row['chose_left']}")
    print()

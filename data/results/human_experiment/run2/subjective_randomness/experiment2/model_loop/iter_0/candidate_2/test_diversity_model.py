import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

def extract_features(s):
    if not s: return 0,0,0,0
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
    
    c1, c2, c3 = counts[1], counts[2], counts[3]
    R = c1 + c2 + c3
    div = math.lgamma(R + 1) - math.lgamma(c1 + 1) - math.lgamma(c2 + 1) - math.lgamma(c3 + 1)
    return c1, c2, c3, div

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/responses.csv')
X = []
y = []
for i, row in df.iterrows():
    sa, sb = row['sequence_a'], row['sequence_b']
    fa = extract_features(sa)
    fb = extract_features(sb)
    diff = [a - b for a, b in zip(fa, fb)]
    X.append(diff)
    y.append(row['chose_left'])

clf = LogisticRegression(fit_intercept=True)
clf.fit(X, y)
print("Coefficients:", clf.coef_)
print("Accuracy:", clf.score(X, y))

import pandas as pd
import numpy as np

df = pd.read_csv("/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv")
imb_a = df["imbalance_a"]
imb_b = df["imbalance_b"]
alts_a = df["alts_a"]
alts_b = df["alts_b"]
p_alts_a = df["p_alts_a"]
p_alts_b = df["p_alts_b"]

chose_left = df["chose_left"]

# Linear vs Quad for alts
theta = 0.6
diff_a = p_alts_a - theta
diff_b = p_alts_b - theta

lin_pen_a = np.abs(diff_a)
lin_pen_b = np.abs(diff_b)
quad_pen_a = diff_a ** 2
quad_pen_b = diff_b ** 2

score_lin_a = -lin_pen_a
score_lin_b = -lin_pen_b
print("corr score_lin_a vs chose_left:", np.corrcoef(score_lin_a, chose_left)[0, 1])

score_quad_a = -quad_pen_a
score_quad_b = -quad_pen_b
print("corr score_quad_a vs chose_left:", np.corrcoef(score_quad_a, chose_left)[0, 1])


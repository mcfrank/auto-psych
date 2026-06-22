import numpy as np
from scipy.special import gammaln

def log_binom(k, n, p):
    # k and n can be arrays
    comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return comb + k * np.log(p) + (n - k) * np.log(1 - p)

n = 20
p_alt = 0.6
for alts in range(20):
    print(f"alts={alts}: {log_binom(alts, 19, p_alt):.2f}")

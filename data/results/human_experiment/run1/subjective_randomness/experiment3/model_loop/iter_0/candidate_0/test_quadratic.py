def penalty_sq_direct(runs, c):
    return sum((r - c)**2 for r in runs)

def penalty_sq_formula(S2, N, Nr, c):
    return S2 - 2 * c * N + (c**2) * Nr

runs = [5, 1, 1, 1]
S2 = sum(r**2 for r in runs)
N = sum(runs)
Nr = len(runs)
c = 1.5

print(f"Direct: {penalty_sq_direct(runs, c)}")
print(f"Formula: {penalty_sq_formula(S2, N, Nr, c)}")

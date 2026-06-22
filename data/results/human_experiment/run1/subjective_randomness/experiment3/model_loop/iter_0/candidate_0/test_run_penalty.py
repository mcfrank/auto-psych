def run_penalty(runs, ideal_L):
    return sum(abs(r - ideal_L) for r in runs)

A = [5, 1, 1, 1]  # length 8, 4 runs
B = [2, 2, 2, 2]  # length 8, 4 runs
print(f"A ({A}): {run_penalty(A, 1.5)}")
print(f"B ({B}): {run_penalty(B, 1.5)}")

print(f"Over-alt (1,1,1,1,1,1): {run_penalty([1]*6, 1.5)}")
print(f"Under-alt (6): {run_penalty([6], 1.5)}")
print(f"Ideal (2,1,2,1): {run_penalty([2,1,2,1], 1.5)}")
def run_penalty_sq(runs, ideal_L):
    return sum((r - ideal_L)**2 for r in runs)

print("\nQuadratic:")
print(f"A ({A}): {run_penalty_sq(A, 1.5)}")
print(f"B ({B}): {run_penalty_sq(B, 1.5)}")
print(f"Over-alt (1,1,1,1,1,1): {run_penalty_sq([1]*6, 1.5)}")
print(f"Under-alt (6): {run_penalty_sq([6], 1.5)}")
print(f"Ideal (2,1,2,1): {run_penalty_sq([2,1,2,1], 1.5)}")

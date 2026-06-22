import numpy as np

def expected_max_run(n):
    # just simulate
    runs = []
    for _ in range(10000):
        seq = np.random.randint(0, 2, n)
        # find max run
        max_r = 1
        current_r = 1
        for i in range(1, n):
            if seq[i] == seq[i-1]:
                current_r += 1
                if current_r > max_r: max_r = current_r
            else:
                current_r = 1
        runs.append(max_r)
    return np.mean(runs)

for n in [2, 3, 4, 5, 10, 20, 50, 100]:
    sim = expected_max_run(n)
    form = np.log(n) / np.log(2.0)
    print(f"n={n}: sim={sim:.2f}, form={form:.2f}")

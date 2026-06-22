def get_run_counts(s):
    runs = []
    current_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    c1 = sum(1 for r in runs if r == 1)
    c2 = sum(1 for r in runs if r == 2)
    c3_plus = sum(1 for r in runs if r >= 3)
    return c1, c2, c3_plus

print(get_run_counts("HTHTHTHT")) # 8 ones
print(get_run_counts("HHTTHHTT")) # 4 twos
print(get_run_counts("HHHHHTTT")) # 2 bigs
print(get_run_counts("HTTHHTTH")) # 2 ones, 3 twos

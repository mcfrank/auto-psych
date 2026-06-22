import math
def run_entropy(s):
    runs = []
    current_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    counts = {}
    for r in runs:
        # group 3 and above
        k = r if r <= 2 else 3
        counts[k] = counts.get(k, 0) + 1
    
    R = sum(counts.values())
    ent = 0
    for k, v in counts.items():
        if v > 0:
            p = v / R
            ent -= p * math.log(p)
    return ent, ent * R

print("HTHTHTHT (8) :", run_entropy("HTHTHTHT")) # 8 ones
print("HHTTHHTT (8) :", run_entropy("HHTTHHTT")) # 4 twos
print("HHHHHTTT (8) :", run_entropy("HHHHHTTT")) # 2 bigs
print("HTTHHTTH (8) :", run_entropy("HTTHHTTH")) # 2 ones, 3 twos
print("HHTHTTHH (8) :", run_entropy("HHTHTTHH")) # 1 two, 1 one, 1 one, 1 two, 1 two -> wait

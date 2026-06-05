def factor_biased_search(stimulus, response_options):
    """Greedy search biased toward producing factors of the target (human multiplication preference)."""
    import json, random, itertools, math

    def generate_successors(nums):
        result = []
        for i, j in itertools.combinations(range(len(nums)), 2):
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
            ops = [(a + b,), (a * b,), (b - a,), (a - b,)]
            if b != 0 and a % b == 0:
                ops.append((a // b,))
            if a != 0 and b % a == 0:
                ops.append((b // a,))
            for (r,) in ops:
                if isinstance(r, int) and r >= 0:
                    result.append(sorted(rest + [r]))
        return result

    def factor_heuristic(nums, target):
        # Favor states where any number is a factor or multiple of target
        factors = [i for i in range(1, target + 1) if target % i == 0]
        score = 0
        for n in nums:
            min_dist = min(abs(n - f) for f in factors)
            score += min_dist
        return score

    def softmax_weights(heuristics, temperature=4.0):
        neg = [-h / temperature for h in heuristics]
        max_neg = max(neg)
        exps = [math.exp(v - max_neg) for v in neg]
        total = sum(exps)
        return [e / total for e in exps]

    def simulate_once(nums, target, max_steps=12):
        state = list(nums)
        for _ in range(max_steps):
            if len(state) == 1:
                return state[0] == target
            successors = generate_successors(state)
            if not successors:
                return False
            heuristics = [factor_heuristic(s, target) for s in successors]
            weights = softmax_weights(heuristics)
            state = random.choices(successors, weights=weights, k=1)[0]
        return len(state) == 1 and state[0] == target

    choices = json.loads(stimulus[0])
    target = int(float(stimulus[1]))
    N = 50
    solved = sum(1 for _ in range(N) if simulate_once(list(choices), target))
    p = solved / N
    return {"left": p, "right": 1.0 - p}

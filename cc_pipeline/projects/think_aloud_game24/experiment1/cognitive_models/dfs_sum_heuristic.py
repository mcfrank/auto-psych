def dfs_sum_heuristic(stimulus, response_options):
    """Stochastic DFS: softmax-weighted node selection by sum-distance to target."""
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

    def sum_heuristic(nums, target):
        return sum(abs(n - target) for n in nums)

    def softmax_weights(heuristics, temperature=5.0):
        neg = [-h / temperature for h in heuristics]
        max_neg = max(neg)
        exps = [math.exp(v - max_neg) for v in neg]
        total = sum(exps)
        return [e / total for e in exps]

    def dfs_once(nums, target, max_depth=12):
        stack = [(list(nums), 0)]
        while stack:
            state, depth = stack.pop()
            if len(state) == 1:
                if state[0] == target:
                    return True
                continue
            if depth >= max_depth:
                continue
            successors = generate_successors(state)
            if not successors:
                continue
            heuristics = [sum_heuristic(s, target) for s in successors]
            weights = softmax_weights(heuristics)
            # shuffle in heuristic-weighted order for stochastic DFS
            order = random.choices(range(len(successors)), weights=weights, k=len(successors))
            seen = set()
            ordered = []
            for idx in order:
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
            for idx in reversed(ordered):
                stack.append((successors[idx], depth + 1))
        return False

    choices = json.loads(stimulus[0])
    target = int(float(stimulus[1]))
    N = 50
    solved = sum(1 for _ in range(N) if dfs_once(list(choices), target))
    p = solved / N
    return {"left": p, "right": 1.0 - p}

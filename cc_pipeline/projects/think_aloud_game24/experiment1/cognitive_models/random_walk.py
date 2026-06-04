def random_walk(stimulus, response_options):
    """Baseline: uniformly random selection of pairs and operations."""
    import json, random, itertools

    def generate_successors(nums):
        result = []
        for i, j in itertools.combinations(range(len(nums)), 2):
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
            ops = [a + b, a * b, b - a, a - b]
            if b != 0 and a % b == 0:
                ops.append(a // b)
            if a != 0 and b % a == 0:
                ops.append(b // a)
            for r in ops:
                if isinstance(r, int) and r >= 0:
                    result.append(sorted(rest + [r]))
        return result

    def simulate_once(nums, target):
        state = list(nums)
        while len(state) > 1:
            successors = generate_successors(state)
            if not successors:
                return False
            state = random.choice(successors)
        return len(state) == 1 and state[0] == target

    choices = json.loads(stimulus[0])
    target = int(float(stimulus[1]))
    N = 50
    solved = sum(1 for _ in range(N) if simulate_once(list(choices), target))
    p = solved / N
    return {"left": p, "right": 1.0 - p}

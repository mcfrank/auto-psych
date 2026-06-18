import json
import random
import itertools

def generate_sequences():
    # lengths 4 to 8
    candidates = []
    
    # Let's generate a diverse set of candidates.
    # To avoid bias towards length 8, let's pick length uniformly, then sequence uniformly.
    
    random.seed(42)
    
    def get_random_seq():
        length = random.randint(4, 8)
        return "".join(random.choices(["H", "T"], k=length))
        
    for _ in range(300):
        a = get_random_seq()
        b = get_random_seq()
        while a == b:
            b = get_random_seq()
        candidates.append({"sequence_a": a, "sequence_b": b})
        
    with open("/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/design/candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

if __name__ == "__main__":
    generate_sequences()

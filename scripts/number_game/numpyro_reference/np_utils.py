"""
Simpler concept statistics that don't require jax

"""
import numpy as np


def concept_set_to_likelihood(concept_set, epsilon=.1):
    C_size = len(concept_set)
    ints = np.arange(1, 101)

    indicator = np.array([ints == x for x in concept_set]).sum(axis=0)

    return indicator * (1 / C_size) * (1 - epsilon)   +   (epsilon / 100)


def concept_set_to_predict(concept_set):
    ints = np.arange(1, 101)
    indicator = np.array([ints == x for x in concept_set], dtype=float).sum(axis=0)
    return indicator

# def concept_set_to_predict(concept_set, epsilon=.05):
#     ints = np.arange(1, 101)
#     indicator = np.array([ints == x for x in concept_set], dtype=float).sum(axis=0)
#     return indicator.clip(min=epsilon, max=1-epsilon)



def get_concept_sets():
    rule_concept_sets = [
        # Powers of n
        [2, 4, 8, 16, 32, 64],
        [3, 9, 27, 81],
        [4, 16, 64],
        [5, 25],
        [6, 36],
        [7, 49],
        [8, 64],
        [9, 81],
        [10, 100],

        # Multiples of n
        [i for i in range(1, 101) if i % 3 == 0],
        [i for i in range(1, 101) if i % 4 == 0],
        [i for i in range(1, 101) if i % 5 == 0],
        [i for i in range(1, 101) if i % 6 == 0],
        [i for i in range(1, 101) if i % 7 == 0],
        [i for i in range(1, 101) if i % 8 == 0],
        [i for i in range(1, 101) if i % 9 == 0],
        [i for i in range(1, 101) if i % 10 == 0],
        [i for i in range(1, 101) if i % 11 == 0],
        [i for i in range(1, 101) if i % 12 == 0],

        # Ends with digit n
        ## [i for i in range(1, 101) if i % 10 == 0],
        [i for i in range(1, 101) if i % 10 == 1],
        [i for i in range(1, 101) if i % 10 == 2],
        [i for i in range(1, 101) if i % 10 == 3],
        [i for i in range(1, 101) if i % 10 == 4],
        [i for i in range(1, 101) if i % 10 == 5],
        [i for i in range(1, 101) if i % 10 == 6],
        [i for i in range(1, 101) if i % 10 == 7],
        [i for i in range(1, 101) if i % 10 == 8],
        [i for i in range(1, 101) if i % 10 == 9],

        # Contains digit n
        ## [i for i in range(1, 101) if '0' in str(i)],
        [i for i in range(1, 101) if '1' in str(i)],
        [i for i in range(1, 101) if '2' in str(i)],
        [i for i in range(1, 101) if '3' in str(i)],
        [i for i in range(1, 101) if '4' in str(i)],
        [i for i in range(1, 101) if '5' in str(i)],
        [i for i in range(1, 101) if '6' in str(i)],
        [i for i in range(1, 101) if '7' in str(i)],
        [i for i in range(1, 101) if '8' in str(i)],
        [i for i in range(1, 101) if '9' in str(i)],

        # is Prime
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
        ### [i for i in range(1, 101) if not any([i == (j1 * j2) for j1 in range(1, i) for j2 in range(1, i)])],

        # is even/odd
        [i for i in range(1, 101) if i % 2 == 0],
        [i for i in range(1, 101) if i % 2 == 1],
        # squares/cubes
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        [1, 8, 27, 64],
    ]

    interval_concept_sets = [
        list(range(i, j+1)) for i in range(1, 101) for j in range(1, 101) if i <= j
    ]

    noise_set = list(range(1, 101))

    return rule_concept_sets, interval_concept_sets, noise_set


def get_likelihoods_predictives(epsilon=.1, noise=False, y_prior=None):
    # Likelihoods  p(X | h)  for each h
    rule_concept_sets, interval_concept_sets, noise_set = get_concept_sets()
    rule_likes = [concept_set_to_likelihood(C, epsilon=epsilon) for C in rule_concept_sets]
    int_likes  = [concept_set_to_likelihood(C, epsilon=epsilon) for C in interval_concept_sets]
    h_likelihoods = rule_likes + int_likes

    # Predictives  p(y | h)
    rule_ylikes = [concept_set_to_predict(C) for C in rule_concept_sets]
    int_ylikes  = [concept_set_to_predict(C) for C in interval_concept_sets]
    y_predictives = rule_ylikes + int_ylikes
    
    noise_like = concept_set_to_likelihood(list(range(1, 101)))
    if noise:
        h_likelihoods  += [noise_like]
        y_predictives += [noise_like]
    if y_prior is not None:
        h_likelihoods  += [noise_like]   # TODO - I think p(X | h) should be uniform, but not p(y | h) ...
        y_predictives += [y_prior]

    return np.array(h_likelihoods), np.array(y_predictives)




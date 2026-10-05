import argparse
import os
import sys

import pickle
import numpy as np
import pandas as pd
import jax.random
import jax.numpy as jnp

import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

PARENT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-4])
sys.path.append(PARENT_PATH)

import src.prob_models.number_game.infer as infer
from src.prob_models.number_game.np_utils import get_likelihoods_predictives




def str2list(s):
    # Subtract 1 to convert [1, 100] to [0, 99] for class indexes
    return [int(i) - 1 for i in s.split(',')]

def X_to_tensor(X_s):
    X_ls = [str2list(s) for s in X_s]

    # Fill missing values in ragged array with -1
    max_x_len = max(len(X) for X in X_ls)
    X_ls = [X + [-1 for _ in range(max_x_len - len(X))] for X in X_ls]

    return jnp.array(X_ls)

def yn_to_tensor(X_s, df, n_support=100):
    X_y_yn = -np.ones((len(X_s), n_support, 2))
    
    for _, row in df.iterrows():
        X, y, yes, no = row['set'], row['target'], row['yes'], row['no']
        X_i = X_s.index(X)

        X_y_yn[X_i, y-1, :] = [yes, yes+no]

    return jnp.array(X_y_yn)


CONCEPT_SETS = [
    # Multiples of 10
    '10,80',
    '10,50,100',
    '80,70,90,100', 
    
    # Multiples of 5
    '90,100,5',
    '90,45',
    '5,65',
    '55,60,25,100',

    # 6x
    '60,54',
    # 7x
    '7,63',
    # 8x
    '8,16',
    '8,32',
    '8,64',

    # 2^x
    '16',
    '16,8', 
    '16,32,2,8',

    '2,8',
    '8',
    '8,16',
    '8,32',
    '8,64',
    '32,4,16',

    # not 2^x but kind of similar (multiples of 4 less than 20)
    '4,16,12',

    # Ends in X
    '81,71,21,31',
    '63,23,43',
    '64,44',
    '64,94,84',
    '55,15,75,45',
    '46,76,66',
    '37,57',
    '97,77,87,47',
    '48,78,38,98',
    '79,59',
    '59,49,99,69',

    # Contains digit X
    '22,62,26,82',  # 2
    '83,38,58,68',  # 8


    # x^2
    '81,25', 
    '4,64,25,81',

    # 3^x   (also 9x)
    '81,9',

    # ???
    '16,54',
    '36,45,54,81',
    '64,96,24,56',  # multiples of 4?
    '20,8,84,100',
    '96,48',

    # Similarity
    '33', 
    '98,83',
    '5,9',
    '33,9',
    '33,5,19',
    '33,17,5,9',

    # Odd number
    '33,3,65',
    '5,77,89',
    '63,27,99,33',
    '63,67',
]


def main(args):

    df_eb = pd.read_csv('./data/stp_number_game/human-data.txt', sep='\t')

    # concept_sets = [
    #     '16',
    #     '16,8', 
    #     '16,32,2,8',
    #     '10,50,100',
    #     '33,17,5,9', 
    # ]
    #

    #### df_eb = df_eb[df_eb['set'].isin(CONCEPT_SETS)].reset_index(drop=True)
    #### X_s  = CONCEPT_SETS
    X_s  = sorted(set(df_eb['set']))

    obs_X = X_to_tensor(X_s)
    obs_y = yn_to_tensor(X_s, df_eb, n_support=100)

    # Likelihoods  p(X | h)  for each h
    h_likelihoods, y_predictives = get_likelihoods_predictives()
    h_likelihoods, y_predictives = jnp.array(h_likelihoods), jnp.array(y_predictives)


    model = infer.number_bpl_model
    mcmc_fname = './data/stp_number_game/mcmc_full.pk'

    # likelihoods p(X | h)  =  \prod_k  p(x_k | h)
    X_ = obs_X[..., None]
    X_likelihoods = dist.Categorical(probs=h_likelihoods).mask(X_ != -1).log_prob(X_).sum(axis=-2)          ##### shape: (n_hypotheses, n_data)


    model_kwargs = dict(
        y_predictives=y_predictives,
        X_likelihoods=X_likelihoods,
        obs_y=obs_y,
    )

    # Print tensor shapes
    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace(**model_kwargs)
    print(numpyro.util.format_shapes(trace))

    kernel = numpyro.infer.NUTS(model)  #, max_tree_depth=20)


    # Run MCMC
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True
    )
    mcmc.run(
        jax.random.PRNGKey(42),
        **model_kwargs
    )
    samples = mcmc.get_samples()
    mcmc.print_summary()

    with open(mcmc_fname, 'wb') as f:
        pickle.dump(mcmc, f)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", default=1000, type=int)
    parser.add_argument("--num-warmup", default=500, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='cpu or gpu')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    # numpyro.set_host_device_count(args.num_chains)
    numpyro.set_host_device_count(args.num_chains)

    main(args)

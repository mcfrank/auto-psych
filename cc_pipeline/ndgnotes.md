

- Because models are preserved across experiments, make a models directory that is top level in the project (rather than separate models per experiment). manifest/registry one shared file that gets extended / probs updated. theorist doesn't need to copy, only add. manifest/registry to directly include paths to py files.



- Theorist to consider residuals and outliers?
- keep track of pre-registered vs posthoc analyses

- theorist should write a small report explaining why model is added 

- design agent writes expt (html+js using jspsych), it can look at prev expt as well as project references?

- maybe switch cogmodels to return response samples (not distributions)?
- allow an inner theory iteration loop on existing data, go to outer experiment loop when there are two models similarly good
- free params in cogmods?

- a lot of parts of cc are specific to subj randomness. need to generalize.
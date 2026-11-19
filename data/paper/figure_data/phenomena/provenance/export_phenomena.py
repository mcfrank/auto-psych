"""Export trial-level data + model predictions for the 'patterns every seed misses' figures.

Reads the analysis intermediates in the session scratchpad (SR per-trial posterior means from the
cached pooled fits, NG MAP grouped-CV predictions, RSA frozen pool predictions) and writes CSVs to
data/paper/figure_data/phenomena/.
"""
import csv, json
import numpy as np, pandas as pd
from pathlib import Path
from pyprojroot import here
from scipy.optimize import minimize

S = Path('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad')
OUT = here('data/paper/figure_data/phenomena'); OUT.mkdir(parents=True, exist_ok=True)
sw = lambda s: sum(x != y for x, y in zip(s, s[1:]))

# ---------------- subjective randomness ----------------
r = pd.read_pickle(S / 'sr/merged.pkl')
resp = pd.read_csv(here('data/results/human_experiment_sr-oct26/analysis/fits/pooled_runs123/responses.csv'))
assert (resp[['sequence_a', 'sequence_b', 'participant_id', 'trial_index', 'chose_left']].values
        == r[['sequence_a', 'sequence_b', 'participant_id', 'trial_index', 'chose_left']].values).all()
stim = pd.read_pickle(S / 'sr/strict3.pkl').set_index(['A', 'B'])   # canonical pair (A < B), CV blend P(choose A)
flip = (r.sequence_a != r.A).values
key = list(zip(r.A, r.B))
blendA = stim.loc[key, 'stack'].values
varA, varB = stim.loc[key, 'A_variety'].values, stim.loc[key, 'B_variety'].values
P = pd.read_pickle(S / 'sr/persons.pkl')
sr = pd.DataFrame({
    'person': r.participant_id, 'trial_index': r.trial_index,
    'sequence_a': r.sequence_a, 'sequence_b': r.sequence_b, 'chose_a': r.chose_left,
    'length': r.sequence_a.str.len(),
    'switches_a': r.sequence_a.map(sw), 'switches_b': r.sequence_b.map(sw),
    'run_variety_a': np.where(flip, varB, varA), 'run_variety_b': np.where(flip, varA, varB),
    'constant_a': (r.sequence_a.map(sw) == 0).astype(int), 'constant_b': (r.sequence_b.map(sw) == 0).astype(int),
})
names = {'ppdg': 'win_run1', 'esp': 'win_run2', 'mstc': 'win_run3',
         'lr': 'seed_local_representativeness', 'ms': 'seed_motif_stack', 'fk': 'seed_falk_konold_dp',
         'feo': 'seed_finite_experience_occurrence'}
for k, v in names.items():
    sr['p_a_' + v] = r[k + '_p'].values
sr['p_a_seed_blend_cv'] = np.where(flip, 1 - blendA, blendA)
sr['streak_floor_pair'] = ((sr.constant_a != sr.constant_b) & (sr.length >= 5)).astype(int)
sr['variety_pair'] = ((sr.switches_a == sr.switches_b) & (sr.run_variety_a != sr.run_variety_b)).astype(int)
sr['person_agree_majority'] = sr.person.map(P.agree_maj)
sr.to_csv(OUT / 'sr_trials.csv', index=False, float_format='%.5g')

# ---------------- number game ----------------
rows = []
for run in (1, 2, 3):
    p = here(f'data/results/number_game_dataset-nov26/collected/run{run}/number_game/experiment3/model_loop/responses.csv')
    with open(p) as f:
        rows += [dict(run=run, **x) for x in csv.DictReader(f)]
ng = pd.DataFrame(rows)
it = pd.read_pickle(S / 'ng/strict.pkl')
it['number_set'] = it.number_set.astype(str); it['target'] = it.target.astype(str)
keep = {'fold': 'cv_fold', 'n_ex': 'n_examples', 'sf_cv': 'p_yes_win_run1', 'nd_cv': 'p_yes_win_run2', 'ew_cv': 'p_yes_win_run3',
        'bpi_cv': 'p_yes_seed_bp_independent', 'bpm_cv': 'p_yes_seed_bp_mixture', 'ten_cv': 'p_yes_seed_tenenbaum',
        'bpi_rc': 'p_yes_seed_bp_independent_relinked', 'bpm_rc': 'p_yes_seed_bp_mixture_relinked',
        'ten_rc': 'p_yes_seed_tenenbaum_relinked', 'stack': 'p_yes_seed_blend_cv'}
it = it[['number_set', 'target'] + list(keep)].rename(columns=keep)
ng = ng.merge(it, on=['number_set', 'target'], how='left', validate='many_to_one')
assert ng.p_yes_win_run1.notna().all() and len(ng) == 22680
ex = ng.number_set.str.split(',').map(lambda v: [int(x) for x in v]); t = ng.target.astype(int)
ng['said_yes'] = ng.rating.astype(float).astype(int)
ng['target_is_example'] = [int(x in e) for e, x in zip(ex, t)]
ng['target_multiple_of_single_example'] = [int(len(e) == 1 and x % e[0] == 0 and x != e[0]) for e, x in zip(ex, t)]
ng['target_factor_of_single_example'] = [int(len(e) == 1 and e[0] % x == 0 and x != e[0]) for e, x in zip(ex, t)]
ng['shared_digit_set'] = [int(len(e) >= 2 and len({v % 10 for v in e}) == 1) for e in ex]
ng['target_keeps_digit'] = [int(x % 10 == e[0] % 10) for e, x in zip(ex, t)]
ng['shared_parity_set'] = [int(len(e) >= 2 and len({v % 2 for v in e}) == 1) for e in ex]
ng['target_keeps_parity'] = [int(x % 2 == e[0] % 2) for e, x in zip(ex, t)]
g = [int(np.gcd.reduce(e)) for e in ex]
ng['common_divisor_set'] = [int(len(e) >= 2 and gg >= 3) for e, gg in zip(ex, g)]
ng['target_keeps_divisor'] = [int(x % gg == 0) for x, gg in zip(t, g)]
ng['distance_outside_range'] = [max(0, min(e) - x, x - max(e)) for e, x in zip(ex, t)]
ng = ng.drop(columns=['rating'])
ng.to_csv(OUT / 'ng_trials.csv', index=False, float_format='%.5g')

# ---------------- RSA (pre-registered experiment) ----------------
F = json.load(open(here('data/rsa/live/finals_pool_preds_final.json'))); B = json.load(open(here('data/rsa/live/bar_pool_preds.json')))
pool = F['pool']; assert pool['digest'] == B['pool']['digest']
off = np.r_[0, np.cumsum(pool['n_entries'])]
M = {**{k: (v['probs'] if isinstance(v, dict) else v) for k, v in F['models'].items()},
     **{k: (v['probs'] if isinstance(v, dict) else v) for k, v in B['models'].items()}}
FIN = list(F['models']); SEEDS = list(B['models'])
col = {'boundary_avoidable_ambiguity_listener': 'final_chain0', 'competition_endpoint_binding_lapse_l2': 'final_chain2',
       'feature_surprisal_ensemble_listener': 'final_chain1'}
col.update({m: m.replace('seed:', 'seed_').replace('promoted:', 'promoted_') for m in SEEDS})
key = {}
for i, c in enumerate(pool['contexts']):
    key.setdefault((json.dumps(c['objects']), c['utterance']), []).append(i)
ppl = pd.read_csv(here('data/rsa_shared/people.csv')); kept = set(ppl[(ppl.study == 'closed_final') & (ppl.kept == True)].person)
tr = [x for x in csv.DictReader(open(here('data/rsa_shared/trials/closed_final.csv')))
      if x['task'] == 'rsa_choice' and x['phase'] == 'test' and x['is_catch'] == 'False' and x['person'] in kept]

def roles(i):
    c = pool['contexts'][i]; objs = c['objects']; u = c['utterance']; nf = [sum(o) for o in objs]
    if u is None:
        return nf, ['no_word'] * len(objs), False
    mf = [nf[j] for j, o in enumerate(objs) if o[u] == 1]
    out = []
    for j, o in enumerate(objs):
        if o[u] != 1: out.append('no_match')
        elif min(mf) == max(mf): out.append('match_all_same_count')
        elif nf[j] == min(mf): out.append('simplest_match')
        elif nf[j] == max(mf): out.append('richest_match')
        else: out.append('middle_match')
    return nf, out, 'middle_match' in out

recs = []
for x in tr:
    ks = key[(json.dumps(json.loads(x['objects'])), int(x['utterance']) if x['utterance'] not in ('', 'None') else None)]
    i = ks[0]; cls = pool['choice_classes'][i]; ch = int(x['choice']); nf, rl, interior = roles(i)
    rec = dict(person=x['person'], condition=x['condition'], display=i, objects=x['objects'], utterance=x['utterance'],
               word=x['word'], feature_names=x['feature_names'], choice=ch, choice_class=cls[ch],
               chosen_feature_count=nf[ch], chosen_role=rl[ch], interior_display=int(interior))
    for m in FIN + SEEDS:
        rec['p_chosen_' + col[m]] = M[m][off[i] + cls.index(cls[ch])]
    recs.append(rec)
rsa = pd.DataFrame(recs)
assert len(rsa) == 14404
# CV mixture of all 17 starting models (weights fit on 4/5 of displays, applied to the held-out fifth)
Pm = rsa[['p_chosen_' + col[m] for m in SEEDS]].values
ud = np.unique(rsa.display); fd = dict(zip(ud, np.random.default_rng(0).integers(0, 5, len(ud))))
fold = rsa.display.map(fd).values; W = {}
for f in range(5):
    trn = fold != f
    nll = lambda z: -np.log(Pm[trn] @ (np.exp(z - z.max()) / np.exp(z - z.max()).sum()) + 1e-12).sum()
    z = minimize(nll, np.zeros(len(SEEDS)), method='L-BFGS-B').x; w = np.exp(z - z.max()); W[f] = w / w.sum()
rsa['p_chosen_seed_blend_cv'] = np.einsum('ij,ij->i', Pm, np.stack([W[f] for f in fold]))
rsa.to_csv(OUT / 'rsa_trials.csv', index=False, float_format='%.5g')

# one row per display x choice class: observed counts and every model's predicted choice probability
cnt = rsa.groupby(['display', 'choice_class']).size()
drows = []
for i in sorted(rsa.display.unique()):
    c = pool['contexts'][i]; cls = pool['choice_classes'][i]; nf, rl, interior = roles(i); n = int((rsa.display == i).sum())
    for k in dict.fromkeys(cls):
        js = [j for j, cc in enumerate(cls) if cc == k]
        row = dict(display=i, objects=json.dumps(c['objects']), utterance=c['utterance'], interior_display=int(interior),
                   n_trials=n, choice_class=k, member_objects=json.dumps(js), n_members=len(js),
                   feature_count=nf[js[0]], role=rl[js[0]], observed_count=int(cnt.get((i, k), 0)))
        for m in FIN + SEEDS:
            row['p_' + col[m]] = M[m][off[i] + cls.index(k)]
        row['p_seed_blend_cv'] = sum(W[fd[i]][s] * M[m][off[i] + cls.index(k)] for s, m in enumerate(SEEDS))
        drows.append(row)
pd.DataFrame(drows).to_csv(OUT / 'rsa_display_classes.csv', index=False, float_format='%.5g')
print('wrote', sorted(p.name for p in OUT.glob('*.csv')))

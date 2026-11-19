import importlib.util, pandas as pd, numpy as np, math
S='/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/sr/'
P='data/results/human_experiment_sr-oct26/analysis/fits/pooled_runs123/models/'
def load(name):
    spec=importlib.util.spec_from_file_location(name,P+name+'.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mods={k:load(k) for k in ['local_representativeness','falk_konold_dp','finite_experience_occurrence','personal_pattern_detection_gain','edge_streak_primacy_recency_pattern_rule','mild_symmetric_trick_coin_streak_weary']}
st=pd.read_pickle(S+'stim.pkl')
rows=[]
for a,b in zip(st.A,st.B):
    d={}
    for s,suf in ((a,'A'),(b,'B')):
        lr=mods['local_representativeness'].compute_features(s,s); fk=mods['falk_konold_dp'].compute_features(s,s)
        fe=mods['finite_experience_occurrence'].compute_features(s,s); pp=mods['personal_pattern_detection_gain'].compute_features(s,s)
        es=mods['edge_streak_primacy_recency_pattern_rule'].compute_features(s,s+'H' if False else s)
        sm=mods['mild_symmetric_trick_coin_streak_weary']._summary(s)
        n=len(s)
        f=dict(alt=lr['p_alts_a'], msimb=lr['multiscale_imbalance_a'], period=lr['periodicity_a'],
               dp=fk['rep_motifs_a']+2*fk['alt_motifs_a'], logocc=math.log(fe['occ_n20_a']) if fe['occ_n20_a']>0 else -50,
               imb=pp['imbalance_a'], maxrun=pp['maxrun_a'], endrun=pp['endrun_a'], pal=pp['palin_a'], variety=pp['variety_a'],
               lopsided=pp['lopsided_a'], anti=pp['antisym_a'], startrun=mods['personal_pattern_detection_gain'].compute_features(s[::-1],s[::-1])['endrun_a'],
               lmark=sm['lm'], alt2=sm['alt2'], gam=sm['gam'], n=n)
        # esp features
        e=mods['edge_streak_primacy_recency_pattern_rule'].compute_features(s,'H'*n)
        f['span']=e['span_a']
        # triplet variety & rule built computed via diff with a reference
        ref=mods['edge_streak_primacy_recency_pattern_rule'].compute_features(s,s)
        for k,v in f.items(): d[f'{suf}_{k}']=v
    rows.append(d)
F=pd.DataFrame(rows)
# triplet variety and rule_built directly
def trip(s):
    nw=len(s)-2
    return 0.0 if nw<1 else len({s[i:i+3] for i in range(nw)})/min(8,nw)
def rule(s):
    n=len(s)
    if len(set(s))<2: return 0.0
    for p in range(2,n//2+1):
        if all(s[i]==s[i+p] for i in range(n-p)): return 1.0
    sw=s.translate(str.maketrans('HT','TH')); return 1.0 if s==sw[::-1] else 0.0
for suf,col in (('A','A'),('B','B')):
    F[f'{suf}_trip']=[trip(s) for s in st[col]]; F[f'{suf}_rule']=[rule(s) for s in st[col]]
st=pd.concat([st[[c for c in st.columns if not c.startswith(('A_','B_'))]],F],axis=1)
st.to_pickle(S+'stimF.pkl'); print(st.shape); print(st.filter(like='A_').describe().T[['mean','min','max']].round(2))

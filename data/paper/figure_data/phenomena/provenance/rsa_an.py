import pandas as pd, numpy as np, json
df=pd.read_pickle('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/rsa/trials.pkl')
C0='boundary_avoidable_ambiguity_listener'; BAR='promoted:rsa_l2_singleton_distinct_feat_color_l0'; TB='seed:rsa_l1_shared_prior'
F=json.load(open('data/rsa/live/finals_pool_preds_final.json')); pool=F['pool']; off=np.r_[0,np.cumsum(pool['n_entries'])]
def describe(i):
    c=pool['contexts'][i]; objs=c['objects']; u=c['utterance']
    nf=[sum(o) for o in objs]
    if u is None: return dict(n_obj=len(objs), utt=False)
    match=[j for j,o in enumerate(objs) if o[u]==1]
    mf=[nf[j] for j in match]
    dup=len(objs)-len({tuple(o) for o in objs})
    interior=sum(1 for j in match if min(mf)<nf[j]<max(mf))
    return dict(n_obj=len(objs), n_feat=len(objs[0]), utt=True, n_match=len(match), n_match_distinct=len({tuple(objs[j]) for j in match}),
                interior=interior, dup=dup, mf_range=max(mf)-min(mf) if mf else 0, has_empty=int(0 in nf))
g=df.groupby('ctx'); d=pd.DataFrame({'n':g.size(),'half':g.half.first(),'c0':g[C0].sum(),'bar':g[BAR].sum(),'tb':g[TB].sum()})
d['gain']=d.c0-d.bar; d['gain_tb']=d.bar-d.tb
d=d.join(pd.DataFrame([describe(i) for i in d.index],index=d.index))
print('total c0-bar',d.gain.sum().round(1),' bar-textbook',d.gain_tb.sum().round(1),' displays',len(d))
print('share of c0 gain from top 10% displays', (d.gain.nlargest(31).sum()/d.gain.sum()).round(2), '; displays where c0 worse', (d.gain<0).sum(), 'sum', d.gain[d.gain<0].sum().round(1))
for col in ['half','n_match','n_match_distinct','interior','dup','has_empty','n_obj']:
    print(d.groupby(col).agg(displays=('n','size'),trials=('n','sum'),c0_vs_bar=('gain','sum'),bar_vs_textbook=('gain_tb','sum')).round(1).to_string(),'\n')
d.to_pickle('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/rsa/disp.pkl')
print('==== top displays')
B=json.load(open('data/rsa/live/bar_pool_preds.json'))
def probs(m,i,src):
    v=src['models'][m]; v=v['probs'] if isinstance(v,dict) else v; return np.round(v[off[i]:off[i+1]],2).tolist()
for i in list(d.gain.nlargest(8).index)+list(d.gain.nsmallest(3).index):
    c=pool['contexts'][i]; cnt=df[df.ctx==i].cls.value_counts().sort_index().to_dict()
    print(i, 'objs',c['objects'],'utt',c['utterance'],'n',d.n[i],'gain %.1f'%d.gain[i],'| human class counts',cnt,'| classes',pool['choice_classes'][i])
    print('     c0 ',probs(C0,i,F),' bar',probs(BAR,i,B))

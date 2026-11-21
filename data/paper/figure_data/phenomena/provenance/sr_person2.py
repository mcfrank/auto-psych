import h5py, numpy as np, pandas as pd
S='/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/sr/'
r=pd.read_pickle(S+'merged.pkl')
sw=lambda s: sum(x!=y for x,y in zip(s,s[1:]))
r['dsw']=r.sequence_a.map(sw)-r.sequence_b.map(sw)
x=r[r.dsw!=0]; agree=np.where(x.dsw>0,x.chose_left,1-x.chose_left)
pp=pd.Series(agree,index=x.participant_id).groupby(level=0).mean().rename('agree')
# agreement with majority per pair
r['maj']=r.groupby(['A','B']).choseA.transform('mean')>0.5
r['agree_maj']=(r.choseA==r.maj).astype(int)
am=r.groupby('participant_id').agree_maj.mean()
P=pd.DataFrame({'agree_sw':pp,'agree_maj':am})
for w in ['ppdg','esp','mstc']:
    P['g_'+w]=(r[w+'_loo']-r['lr_loo']).groupby(r.participant_id).sum()
P['grp']=pd.cut(P.agree_maj,[0,0.45,0.55,0.65,0.75,1.0])
print(P.groupby('grp').agg(people=('agree_maj','size'),g_ppdg=('g_ppdg','sum'),g_esp=('g_esp','sum'),g_mstc=('g_mstc','sum')).round(0))
print('totals', P[['g_ppdg','g_esp','g_mstc']].sum().round(0).to_dict())
lo=P.agree_maj<0.55
print(f'people agreeing with the majority on <55% of trials: {lo.sum()} ({lo.mean():.0%}); their share of the winner gain: ppdg {P.g_ppdg[lo].sum()/P.g_ppdg.sum():.0%}, esp {P.g_esp[lo].sum()/P.g_esp.sum():.0%}, mstc {P.g_mstc[lo].sum()/P.g_mstc.sum():.0%}')
P.to_pickle(S+'persons.pkl')

import pandas as pd, numpy as np
d=pd.read_pickle('sr/stimF.pkl').copy()
s=pd.read_pickle('sr/stim.pkl')
for c in ['A_sw','B_sw','A_nruns','B_nruns','A_maxrun','B_maxrun']: d[c]=s[c]
d['altA']=d.A_sw/(d.len-1); d['altB']=d.B_sw/(d.len-1)
# orient: X = the sequence with more switches (ties: keep A)
flip=d.B_sw>d.A_sw
def o(a,b): return np.where(flip,d[b],d[a])
for m in ['ppdg','esp','mstc','lr','obs']: d['p_'+m]=np.where(flip,1-d[m],d[m])
d['altX']=o('altA','altB'); d['altY']=o('altB','altA')
d['runX']=o('A_maxrun','B_maxrun'); d['runY']=o('B_maxrun','A_maxrun')
d['varX']=o('A_variety','B_variety'); d['varY']=o('B_variety','A_variety')
d['symX']=o('A_pal','B_pal')+o('A_anti','B_anti')+o('A_rule','B_rule'); d['symY']=o('B_pal','A_pal')+o('B_anti','A_anti')+o('B_rule','A_rule')
d['imbX']=o('A_lopsided','B_lopsided'); d['imbY']=o('B_lopsided','A_lopsided')
for m in ['ppdg','esp','mstc']: d['g_'+m]=d['ll_'+m]-d['ll_lr']
def row(name,m):
    x=d[m]; w=x.n
    r=dict(pattern=name,pairs=int(m.sum()),trials=int(w.sum()))
    for k in ['obs','lr','ppdg','esp','mstc']: r[k]=np.average(x['p_'+k],weights=w)
    for k in ['ppdg','esp','mstc']: r['gain_'+k]=x['g_'+k].sum()
    return r
L=d.len
rows=[
 row('Y is all one side (HHHH..)', (d.altY==0)&(d.altX>0)),
 row('Y has one switch only', (d.A_sw.where(~flip,d.B_sw).pipe(lambda z: o('B_sw','A_sw'))==1)),
 row('X perfectly alternates (HTHT..)', d.altX==1),
 row('X over-alternates (0.7<alt<1)', (d.altX>0.7)&(d.altX<1)),
 row('X moderate (0.4-0.7), Y streaky (<0.4)', (d.altX.between(0.4,0.7))&(d.altY<0.4)),
 row('equal #switches', d.A_sw==d.B_sw),
 row('equal #switches, X more run variety', (d.A_sw==d.B_sw)&(d.varX!=d.varY)),
 row('equal #switches, one is symmetric/rule-built', (d.A_sw==d.B_sw)&((d.symX>0)!=(d.symY>0))),
]
t=pd.DataFrame(rows); pd.set_option('display.width',250); print(t.round(3).to_string(index=False))
print('total gain',[d['g_'+m].sum().round(0) for m in ['ppdg','esp','mstc']])
# equal-switch pairs: choice vs variety and symmetry
e=d[d.A_sw==d.B_sw].copy()
e['dvar']=e.A_variety-e.B_variety; e['dsym']=(e.A_pal+e.A_anti+e.A_rule)-(e.B_pal+e.B_anti+e.B_rule); e['dlop']=e.A_lopsided-e.B_lopsided
for c in ['dvar','dsym','dlop']:
    for sgn,mm in [('>0',e[c]>0),('<0',e[c]<0)]:
        x=e[mm]; w=x.n
        if len(x): print(c,sgn,len(x),'obs',round(np.average(x.obs,weights=w),3),'lr',round(np.average(x.lr,weights=w),3),'ppdg',round(np.average(x.ppdg,weights=w),3),'esp',round(np.average(x.esp,weights=w),3),'mstc',round(np.average(x.mstc,weights=w),3))
print('\n--- X = perfect alternation, binned by the other sequence Y alternation rate')
x=d[d.altX==1].copy(); x['bin']=pd.cut(x.altY,[-.01,.2,.4,.55,.7,.99])
print(x.groupby('bin',observed=True).apply(lambda g: pd.Series({'pairs':len(g),'obs':np.average(g.p_obs,weights=g.n),'lr':np.average(g.p_lr,weights=g.n),'ppdg':np.average(g.p_ppdg,weights=g.n),'esp':np.average(g.p_esp,weights=g.n),'mstc':np.average(g.p_mstc,weights=g.n)})).round(3))
print('\n--- Y all one side: binned by X alternation')
x=d[(d.altY==0)&(d.altX>0)].copy(); x['bin']=pd.cut(x.altX,[0,.2,.4,.6,.8,1.0])
print(x.groupby('bin',observed=True).apply(lambda g: pd.Series({'pairs':len(g),'obs':np.average(g.p_obs,weights=g.n),'lr':np.average(g.p_lr,weights=g.n),'ppdg':np.average(g.p_ppdg,weights=g.n),'esp':np.average(g.p_esp,weights=g.n),'mstc':np.average(g.p_mstc,weights=g.n)})).round(3))
print('\n--- examples, equal switches, largest |obs-lr|')
e=d[d.A_sw==d.B_sw].copy(); e['err']=(e.obs-e.lr).abs()
print(e.sort_values('err',ascending=False).head(10)[['A','B','n','obs','lr','ppdg','esp','mstc','A_variety','B_variety','A_lopsided','B_lopsided']].round(2).to_string(index=False))

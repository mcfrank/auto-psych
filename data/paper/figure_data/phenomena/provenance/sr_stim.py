import pandas as pd, numpy as np
S='/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/sr/'
r=pd.read_pickle(S+'merged.pkl'); M=['ppdg','esp','mstc','lr','ms','fk','feo']
g=r.groupby(['A','B'])
st=g.agg(n=('choseA','size'),k=('choseA','sum'),**{m:(m+'_pA','mean') for m in M}).reset_index()
st['obs']=st.k/st.n
def bll(p,k,n): p=np.clip(p,1e-9,1-1e-9); return k*np.log(p)+(n-k)*np.log(1-p)
for m in M: st['ll_'+m]=bll(st[m],st.k,st.n)
def feats(s):
    n=len(s); sw=sum(x!=y for x,y in zip(s,s[1:]))
    runs=[];c=1
    for x,y in zip(s,s[1:]):
        if x==y: c+=1
        else: runs.append(c); c=1
    runs.append(c)
    t=lo=hi=0
    for ch in s: t+=1 if ch=='H' else -1; lo=min(lo,t); hi=max(hi,t)
    comp=''.join('T' if ch=='H' else 'H' for ch in s[::-1])
    per=any(all(s[i]==s[i+p] for i in range(n-p)) for p in range(1,n//2+1)) and len(set(s))>1
    return dict(sw=sw, heads=s.count('H'), imb=abs(2*s.count('H')-n), maxrun=max(runs), endrun=runs[-1], startrun=runs[0],
                span=hi-lo, pal=int(s==s[::-1]), anti=int(s==comp), periodic=int(per), nruns=len(runs), variety=len(set(runs)))
FA=pd.DataFrame([feats(s) for s in st.A]); FB=pd.DataFrame([feats(s) for s in st.B])
for c in FA: st['A_'+c]=FA[c]; st['B_'+c]=FB[c]
st['len']=st.A.str.len()
st.to_pickle(S+'stim.pkl')
pd.set_option('display.width',250)
for w in ['ppdg','esp','mstc']:
    st['gain']=st['ll_'+w]-st['ll_lr']
    print(f'\n=== {w}: stim-level gain over LR {st.gain.sum():.0f} nats; by length:')
    print(st.groupby('len').agg(pairs=('n','size'),trials=('n','sum'),gain=('gain','sum')).round(0).T.to_string())
    cols=['A','B','n','obs','lr',w,'gain']
    print(st.sort_values('gain',ascending=False)[cols].head(18).round(2).to_string(index=False))
    print('  worst:'); print(st.sort_values('gain')[cols].head(6).round(2).to_string(index=False))

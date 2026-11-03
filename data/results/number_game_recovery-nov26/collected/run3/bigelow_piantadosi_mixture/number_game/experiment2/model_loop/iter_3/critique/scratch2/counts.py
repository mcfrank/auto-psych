import pandas as pd, numpy as np
d=pd.read_csv('../../../responses.csv')
P=set(n for n in range(2,101) if all(n%a for a in range(2,int(n**.5)+1)))
sets={s:tuple(int(x) for x in s.split(',')) for s in d.number_set.unique()}
t=d.target.values; s=d.number_set.values
def cnt(name, f):
    m=np.array([f(sets[a],b) for a,b in zip(s,t)]); print(name, m.sum(), d.rating[m].mean() if m.any() else None)
cnt('prime sets prime tgt', lambda e,x: len(e)>=2 and all(v in P for v in e) and x not in e and x in P)
cnt('prime sets oddcomp tgt', lambda e,x: len(e)>=2 and all(v in P for v in e) and x not in e and x%2==1 and x not in P and x>1)
cnt('tight far', lambda e,x: len(e)>=2 and max(e)-min(e)<=20 and min(abs(x-v) for v in e)>=30)
cnt('par-in-hull-violate', lambda e,x: len(e)>=2 and len(set(v%2 for v in e))==1 and min(e)<x<max(e) and x%2!=e[0]%2)
cnt('par-out-hull-match', lambda e,x: len(e)>=2 and len(set(v%2 for v in e))==1 and (x<min(e)-10 or x>max(e)+10) and x%2==e[0]%2)
cnt('enddig tens tgt', lambda e,x: len(e)>=2 and len(set(v%10 for v in e))==1 and x>=10 and x//10==e[0]%10 and x%10!=e[0]%10)
cnt('enddig other', lambda e,x: len(e)>=2 and len(set(v%10 for v in e))==1 and x not in e and x%10!=e[0]%10 and not (x>=10 and x//10==e[0]%10))
cnt('single mult/div', lambda e,x: len(e)==1 and x!=e[0] and (x%e[0]==0 or e[0]%x==0))
cnt('single other', lambda e,x: len(e)==1 and x!=e[0] and not (x%e[0]==0 or e[0]%x==0) and x%2==e[0]%2)

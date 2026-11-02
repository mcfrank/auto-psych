import glob, time, numpy as np, pandas as pd, math, importlib.util
d=pd.read_csv('../../../responses.csv')
for f in sorted(glob.glob('../test_stats/*.py')):
    ns={'np':np,'pd':pd,'math':math}; exec(open(f).read(), ns)
    t0=time.time(); v=ns['test_statistic'](d); d2=d.copy(); d2['rating']=np.random.randint(0,2,len(d)); v2=ns['test_statistic'](d2)
    print(f.split('/')[-1], round(v,4), round(v2,4), round(time.time()-t0,3))

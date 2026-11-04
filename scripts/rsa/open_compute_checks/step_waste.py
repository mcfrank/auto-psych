import glob,json,os,sys
import numpy as np, h5py
res=[]
fs=sorted(glob.glob(sys.argv[1]+"/.fit_cache/*.nc"))
for f in fs[::3]:
    try:
        with h5py.File(f,"r") as d:
            s=np.asarray(d["sample_stats/n_steps"]).astype(float)
    except Exception as e:
        print("skip",type(e).__name__); continue
    if s.ndim!=2: continue
    res.append((s.shape[1], s.max(0).sum()/s.sum(1).max(), s.sum()/s.max(0).sum(), s.mean(), os.path.getsize(f)/2**20))
for dr in sorted({r[0] for r in res}):
    R=[r for r in res if r[0]==dr]
    m=lambda i: round(float(np.median([r[i] for r in R])),2)
    print("draws per chain",dr,"fits",len(R),"| slowest-chain speedup if parallel",m(1),"| mean chains active per step (of 4)",m(2),"| steps per draw",m(3),"| nc MB",m(4))

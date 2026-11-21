import json, csv, numpy as np, pandas as pd, collections
F=json.load(open('data/rsa/live/finals_pool_preds_final.json')); B=json.load(open('data/rsa/live/bar_pool_preds.json'))
pool=F['pool']; assert pool['digest']==B['pool']['digest']
off=np.r_[0,np.cumsum(pool['n_entries'])]
key={}
for i,c in enumerate(pool['contexts']):
    k=(json.dumps(c['objects']), c['utterance'], c.get('valence'), c.get('familiarization'), c.get('grayscale'))
    key.setdefault((json.dumps(c['objects']), c['utterance']),[]).append(i)
models={**F['models'], **B['models']}
print({k:(type(v).__name__, len(v) if hasattr(v,'__len__') else v) for k,v in list(models.items())[:2]})
P=json.load(open('data/rsa/live/../people.csv')) if False else None
ppl=pd.read_csv('data/rsa_shared/people.csv'); kept=set(ppl[(ppl.study=='closed_final')&(ppl.kept==True)].person)
rows=[r for r in csv.DictReader(open('data/rsa_shared/trials/closed_final.csv')) if r['task']=='rsa_choice' and r['phase']=='test' and r['is_catch']=='False' and r['person'] in kept]
print('trials',len(rows))
amb=collections.Counter(); out=[]
for r in rows:
    ks=key.get((json.dumps(json.loads(r['objects'])), int(r['utterance']) if r['utterance'] not in ('','None') else None))
    if not ks: amb['nomatch']+=1; continue
    if len(ks)>1: amb['multi']+=1
    i=ks[0]; cls=pool['choice_classes'][i]; ch=int(r['choice'])
    first=cls.index(cls[ch])
    rec=dict(ctx=i, label=r['condition'], half=r['condition'].split('_')[0], cls=cls[ch], objects=r['objects'], word=r['word'], item=r['item'], fnames=r['feature_names'])
    for m,v in models.items():
        p=v[off[i]+first] if isinstance(v,list) else v['probs'][off[i]+first]
        rec[m]=np.log(max(p,1e-12))
    out.append(rec)
print(amb, len(out))
df=pd.DataFrame(out); df.to_pickle('/private/tmp/claude-503/-Users-bp3902-Documents-auto-psych/0d71aa57-486d-4f04-9d8f-23bd23e237af/scratchpad/rsa/trials.pkl')
print(df[list(models)].sum().sort_values(ascending=False).round(1).to_string())

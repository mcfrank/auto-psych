"""Where a loop's compute goes, from the chain's sidecars and ledger: aggregates only (no trials)."""
import collections, glob, json, os, sys, statistics as st
root = sys.argv[1]
for exp in ("experiment1", "experiment2"):
    loop = f"{root}/{exp}/loop"
    rows = []
    for f in glob.glob(f"{loop}/.fit_cache/*.json"):
        if f.endswith(".warmup.json"): continue
        try: m = json.load(open(f))
        except Exception: continue
        t = m.get("timing") or {}
        if "warmup_sec" not in t: continue
        rows.append(dict(name=m["theory_name"], w=t["warmup_sec"], s=t["sampling_sec"], ws=t.get("warmup_steps", 0), ss=t.get("sampling_steps", 0),
                         full=t.get("warmup_steps", 0) > 0 and t["warmup_sec"] > 0, bad=bool(m.get("convergence_problems"))))
    # full fits have 1000+1000 iterations: the longer warmup per fit of its theory
    by = collections.defaultdict(list)
    for r in rows: by[r["name"]].append(r)
    full, fold = [], []
    for n, rs in by.items():
        rs.sort(key=lambda r: -(r["ws"]))
        if len(rs) >= 2: full.append(rs[0]); fold += rs[1:]
        else: full.append(rs[0])
    h = lambda L, k: sum(r[k] for r in L) / 3600
    print(exp, "fits", len(rows), "theories", len(by))
    print(" core-hours: full fits %.0f (warmup %.0f, sampling %.0f); folds %.0f (warmup %.0f, sampling %.0f)" % (h(full,"w")+h(full,"s"), h(full,"w"), h(full,"s"), h(fold,"w")+h(fold,"s"), h(fold,"w"), h(fold,"s")))
    q = lambda v: [round(x) for x in (min(v), st.median(v), sorted(v)[int(.9*len(v))], max(v))] if v else None
    print(" minutes min/median/p90/max: full", q([(r["w"]+r["s"])/60 for r in full]), "folds", q([(r["w"]+r["s"])/60 for r in fold]))
    print(" sec per gradient step median: full %.3f folds %.3f" % (st.median([(r["w"]+r["s"])/max(1,r["ws"]+r["ss"]) for r in full]), st.median([(r["w"]+r["s"])/max(1,r["ws"]+r["ss"]) for r in fold]) if fold else 0))
    print(" steps per iteration, full: warmup %.1f sampling %.1f (4 chains x 1000 each)" % (st.median([r["ws"]/4000 for r in full]), st.median([r["ss"]/4000 for r in full])))
    print(" fits with convergence problems: full %d of %d, folds %d of %d" % (sum(r["bad"] for r in full), len(full), sum(r["bad"] for r in fold), len(fold)))
    led = [json.loads(l) for l in open(f"{loop}/attempted_hypotheses.jsonl") if l.strip()]
    print(" ledger outcomes:", dict(collections.Counter(e.get("outcome") for e in led)))
    why = collections.Counter()
    for e in led:
        if e.get("outcome") == "rejected":
            d = str(e.get("detail", ""))[:40].lower()
            why["too slow" if "minutes" in d or "time" in d or "slow" in d else "novelty" if "novel" in d or "rmse" in d else "other"] += 1
    print(" rejections:", dict(why))
    h2 = json.load(open(f"{loop}/history.json"))
    best = [(s["round"], s["best_model"]) for s in h2]
    print(" steps", len(h2), "incumbents", len({b for _, b in best}))
    last = h2[-1]["standing"]
    fin = sorted((d["sel_diff"] for d in last.values() if d.get("sel_diff") not in (None, float("inf"))))
    print(" last step: live", len(last), "sel_diff quantiles", [round(x) for x in (fin[0], fin[len(fin)//2], fin[-1])] if fin else None, "share >100 nats behind", round(sum(x > 100 for x in fin)/max(1,len(fin)), 2))

"""The design review's §10 tests, on copies of chain 1's files: no agents, no Prolific, nothing written to the chain.
Prints and saves aggregates only (timings, memory, scores by theory), never trials.

  fit <theory> <method>      a full fit in this process: time and memory by stage, posterior against the lab node's fit,
                             and (vectorized) the adapted mass matrix for `fold`
  fold <theory> <k> <warmup> [seed]  one CV fold starting from that adaptation, against the chain's own fold
  baseline <theory> <k> <seed>       the same fold at the chain's own settings with another seed: the noise baseline
  thin <work dir>            standings with each fold cut to its first 500, 250, 125 draws a chain
"""
import json, os, pickle, sys, threading, time
from dataclasses import replace
from pathlib import Path

BASE = Path(os.environ["OPT_BASE"])
INPUT = BASE / "input"
SINGLE = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def status_kb(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return 0


class Timeline:
    """Memory every 20 s, and marks at the fit's stages."""

    def __init__(self):
        self.t0, self.samples, self.marks, self.stop = time.monotonic(), [], [], False
        threading.Thread(target=self.run, daemon=True).start()

    def now(self):
        return round((time.monotonic() - self.t0) / 60, 2)

    def run(self):
        while not self.stop:
            self.samples.append((self.now(), round(status_kb("VmRSS") / 2**20, 2), round(status_kb("VmHWM") / 2**20, 2)))
            time.sleep(20)

    def mark(self, stage):
        self.marks.append(dict(stage=stage, minutes=self.now(), rss_gb=round(status_kb("VmRSS") / 2**20, 2),
                               hwm_gb=round(status_kb("VmHWM") / 2**20, 2)))
        print(json.dumps(self.marks[-1]), flush=True)


def write_out(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    os.replace(tmp, path)
    print(json.dumps({k: v for k, v in out.items() if k not in ("memory", "stages")}, default=float)[:1500], flush=True)


def machine():
    cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    return dict(host=os.uname().nodename, cpu=cpu, partition=os.environ.get("SLURM_JOB_PARTITION"),
                cpus=os.environ.get("SLURM_CPUS_PER_TASK"), restarts=int(os.environ.get("SLURM_RESTART_COUNT", 0)),
                job=f"{os.environ.get('SLURM_ARRAY_JOB_ID')}_{os.environ.get('SLURM_ARRAY_TASK_ID')}")


def full_settings():
    from src.rsa.fit import FitSettings
    return FitSettings(dense_mass=True)


def read_cached(theory_dir, path, corpus, settings, name, init=None):
    from src.rsa_open.loop import fitting as ft
    nc, meta = ft.cache_paths(theory_dir, name, ft.fingerprint(path, corpus, settings, init))
    if not nc.exists():
        raise FileNotFoundError(f"{nc.name}: not the chain's fit (the inputs differ)")
    return ft._read(nc, meta, log_likelihood=False), json.loads(meta.read_text())


def run_fit(name, method):
    os.environ["XLA_FLAGS"] = SINGLE + (" --xla_force_host_platform_device_count=4" if method == "parallel" else "")
    import numpy as np
    import src.rsa_open.corpus as corpus
    from numpyro.infer import MCMC
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.loop import fitting as ft

    td = INPUT / name
    path, corpus_path, settings = td / f"{name}.py", INPUT / "corpus.json", full_settings()
    read_cached(td, path, corpus_path, settings, name)  # fail now, not after the fit, if this isn't the chain's input
    tl = Timeline()
    holder = {}

    def make_mcmc(kernel, **k):
        k["chain_method"] = method
        holder["mcmc"] = MCMC(kernel, **k)
        return holder["mcmc"]

    corpus.MCMC = make_mcmc
    original_ll = corpus.pointwise_log_lik
    ll_sec = [0.0]

    def timed_ll(*a, **k):
        t = time.monotonic()
        tl.mark("log_likelihood block starts")
        out = original_ll(*a, **k)
        ll_sec[0] += time.monotonic() - t
        return out

    corpus.pointwise_log_lik = timed_ll
    tl.mark("start")
    fitted = corpus.fit_corpus(TheoryModel(path, name=name), ft.read_corpus(corpus_path), settings,
                               on_warmup=lambda r: tl.mark("warmup done"))
    tl.mark("fit_corpus done")
    run_dir = BASE / "runs" / f"{name}_{method}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    ft._write(fitted, run_dir / "fit.nc", run_dir / "fit.json", record_peak=True)
    write_sec = time.monotonic() - t
    tl.mark("written")
    ref, ref_meta = read_cached(td, path, corpus_path, settings, name)
    tl.mark("reference read")
    a, b = fitted.flat(), ref.flat()
    diffs = {k: abs(float(np.mean(a[k]) - np.mean(b[k]))) / max(float(np.std(b[k])), 1e-12) for k in b}
    if method == "vectorized":
        state = holder["mcmc"].post_warmup_state.adapt_state
        with open(td / "adaptation.pkl.tmp", "wb") as f:
            pickle.dump(dict(inverse_mass_matrix={k: np.asarray(v) for k, v in state.inverse_mass_matrix.items()},
                             step_size=np.asarray(state.step_size)), f)
        os.replace(td / "adaptation.pkl.tmp", td / "adaptation.pkl")
    tl.stop = True
    write_out(BASE / "results" / f"fit_{name}_{method}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.json", dict(
        test="fit", theory=name, method=method, machine=machine(), timing=fitted.timing, log_lik_min=round(ll_sec[0] / 60, 2),
        write_min=round(write_sec / 60, 2), nc_mb=round((run_dir / "fit.nc").stat().st_size / 2**20),
        peak_gb=round(status_kb("VmHWM") / 2**20, 2), problems=fitted.convergence_problems,
        lab_node=dict(timing=ref_meta["timing"], peak_mb=ref_meta.get("peak_hwm_mb")),
        posterior_mean_max_diff_in_sd=round(max(diffs.values()), 4), stages=tl.marks, memory=tl.samples[::6]))


def run_fold(name, k, warmup, seed=None, reuse=True):
    os.environ["XLA_FLAGS"] = SINGLE
    import numpy as np
    import src.rsa_open.corpus as corpus
    from numpyro.infer import NUTS
    from src.rsa_open.corpus import assign_folds, heldout_lpd
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.loop import fitting as ft

    td = INPUT / name
    path, corpus_path = td / f"{name}.py", INPUT / "corpus.json"
    fold_path = INPUT / "cv" / f"fold_{k}.json"
    base_settings = full_settings()
    full, _ = read_cached(td, path, corpus_path, base_settings, name)
    start = {p: float(np.median(v)) for p, v in full.flat().items()}
    fold_settings = replace(base_settings, num_warmup=300, num_samples=500)
    chain_fold, chain_meta = read_cached(td, path, fold_path, fold_settings, name, start)
    if reuse:
        with open(td / "adaptation.pkl", "rb") as f:
            adapted = pickle.load(f)
        # one matrix for every chain: the mean of the full fit's chains
        mass = {key: np.mean(v, axis=0) for key, v in adapted["inverse_mass_matrix"].items()}
        step = float(np.mean(adapted["step_size"]))

        def make_nuts(model, **kw):
            return NUTS(model, inverse_mass_matrix=mass, step_size=step, adapt_mass_matrix=False, **kw)

        corpus.NUTS = make_nuts
    blocks = ft.read_corpus(corpus_path)
    run_settings = replace(fold_settings, num_warmup=warmup, seed=fold_settings.seed if seed is None else seed)
    t0 = time.monotonic()
    theory = TheoryModel(path, name=name)
    fitted = corpus.fit_corpus(theory, ft.read_corpus(fold_path), run_settings, init=start)
    fit_min = (time.monotonic() - t0) / 60
    scores, pointwise = {}, {}
    for b in blocks:
        rows = np.where(assign_folds(b, 5, seed=base_settings.seed) == k)[0]
        if len(rows):
            sub = b.subset(rows)
            new, old = heldout_lpd(theory, fitted, sub), heldout_lpd(theory, chain_fold, sub)
            scores[b.name] = dict(n=len(rows), new=float(np.sum(new)), chain=float(np.sum(old)),
                                  sd_of_trial_diff=float(np.std(new - old)))
            pointwise.update({f"{b.name}|rows": rows, f"{b.name}|new": np.asarray(new), f"{b.name}|chain": np.asarray(old)})
    d = fitted.diagnostics
    method = f"{'reuse' if reuse else 'chain_settings'}_w{warmup}_s{run_settings.seed}"
    if seed is not None:  # stays on Sherlock: for the clustered SEs, never brought back
        (BASE / "pointwise").mkdir(exist_ok=True)
        np.savez(BASE / "pointwise" / f"{name}__{k}__{method}.npz", **pointwise)
    out_name = f"fold_{name}_{k}_{method}.json" if seed is not None else f"fold_{name}_{k}_w{warmup}.json"
    write_out(BASE / ("results2" if seed is not None else "results") / out_name, dict(
        test="fold", theory=name, fold=k, warmup=warmup, seed=run_settings.seed, reuse=reuse, machine=machine(),
        fit_min=round(fit_min, 1), timing=fitted.timing,
        chain_timing=chain_meta["timing"], problems=fitted.convergence_problems, diagnostics=str(d)[:300],
        peak_gb=round(status_kb("VmHWM") / 2**20, 2), heldout=scores))


def run_thin(work):
    os.environ["XLA_FLAGS"] = SINGLE
    from src.models.pymc_inference import convergence_problems
    from src.rsa.fit import FitSettings
    from src.rsa_open.loop import fitting as ft
    from src.rsa_open.loop import orchestrator as o

    def refuse(*a, **k):
        raise RuntimeError(f"cache miss for {a[1] if len(a) > 1 else a}: this test never fits")

    def no_agents(*a, **k):
        raise RuntimeError("this test starts no agent")

    ft._fit_in_child = refuse
    keep = [None]
    original = o.loop_fit

    def thinned(path, name, corpus_path, settings, cache_dir, **k):
        fit = original(path, name, corpus_path, settings, cache_dir, **k)
        if k.get("init") is None or keep[0] is None:
            return fit
        small = fit.idata.isel(draw=slice(0, keep[0]))
        return replace(fit, idata=small, convergence_problems=convergence_problems(small, fit.param_names))

    o.loop_fit = thinned
    cfg = o.OpenLoopConfig(corpus_path=work / "corpus.json", seed_models_dir=work / "unused", results_dir=work / "loop",
                           settings=FitSettings(dense_mass=True))
    loop = o.OpenLoop(cfg, no_agents)
    loop.resume()
    failed = {n: None for n, v in loop.cv.items() if v is None}
    problems = dict(loop.cv_problems)
    step = loop.history[-1]
    out = dict(test="thin", step=step["step"], recorded_best=step["best_model"], live=len(loop.live), runs={})
    recorded = {n: d.get("sel_elpd") for n, d in step["standing"].items()}
    for k in (500, 250, 125):
        keep[0] = k
        loop.cv, loop.cv_problems = dict(failed), dict(problems)
        t = time.monotonic()
        s = loop.standing()
        scored = {n: d for n, d in s.items() if d["sel_elpd"] > float("-inf")}
        best = loop.incumbent(s)
        out["runs"][str(k)] = dict(
            minutes=round((time.monotonic() - t) / 60, 1), best=best,
            not_trusted=sorted(n for n, d in s.items() if not d.get("trusted")),
            sel_elpd={n: round(d["sel_elpd"], 2) for n, d in scored.items()},
            sel_diff={n: round(d["sel_diff"], 2) for n, d in scored.items()},
            sel_dse={n: round(d["sel_dse"], 2) for n, d in scored.items()},
            max_abs_diff_from_recorded=round(max((abs(d["sel_elpd"] - recorded[n]) for n, d in scored.items()
                                                  if recorded.get(n) not in (None, float("-inf"))), default=0.0), 3))
        write_out(BASE / "results" / "thin.json", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        run_fit(sys.argv[2], sys.argv[3])
    elif cmd == "fold":
        run_fold(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif cmd == "baseline":
        run_fold(sys.argv[2], int(sys.argv[3]), 300, int(sys.argv[4]), reuse=False)
    elif cmd == "thin":
        run_thin(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown test {cmd}")

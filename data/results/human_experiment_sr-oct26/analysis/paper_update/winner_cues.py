"""What drives each run winner's predictions, for the qualitative section.

For each run's winner, on its own run's data (experiment-3 cumulative
responses) and its own-run fit (cache only, as winner_params.py), the typical
participant's decision variable (every person-level offset at 0) is split into
its additive terms, per posterior draw, on every distinct stimulus pair, scored
in the orientation of report_fits.stimulus_frame (the alphabetically first
sequence as "a"). Writes, per model:

- results/winner_cues_<run>.csv: per term, its spread across the stimuli
  (SD of the posterior-mean term, logit units) and the correlation of the
  typical participant's P(choose a) with the people's proportion, with all
  terms and with that term removed;
- results/winner_cue_terms_<run>.csv: per stimulus, every term, the model's
  P(choose a), the people's proportion and count, for picking examples.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
from pyprojroot import here

sys.path.insert(0, str(here()))
SERIES = here() / "data/results/human_experiment_sr-oct26"
S = SERIES / "analysis/paper_update"
OUT = SERIES / "analysis/results"
MAX_DRAWS = 1000


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ph = load("plot_human_fit_runs", S / "plot_human_fit_runs.py")
from src.models.pymc_inference import clear_fit_cache, fit_model  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stimuli(responses: pd.DataFrame) -> pd.DataFrame:
    """Distinct unordered pairs, a = the alphabetically first sequence, with the
    proportion of people choosing a."""
    first = np.where(responses.sequence_a < responses.sequence_b, responses.sequence_a, responses.sequence_b)
    second = np.where(responses.sequence_a < responses.sequence_b, responses.sequence_b, responses.sequence_a)
    chose_first = np.where(responses.sequence_a == first, responses.chose_left, 1 - responses.chose_left)
    df = pd.DataFrame({"a": first, "b": second, "chose_a": chose_first})
    return df.groupby(["a", "b"]).chose_a.agg(people="mean", n="size").reset_index()


def draws(post, var):
    x = post[var].values
    x = x.reshape(-1, *x.shape[2:])
    idx = np.linspace(0, len(x) - 1, min(MAX_DRAWS, len(x))).astype(int)
    return x[idx][:, None]  # draws x 1, broadcasts over stimuli


def run1_terms(mod, post, stim):
    f = pd.DataFrame([mod.compute_features(a, b) for a, b in zip(stim.a, stim.b)])
    v = {k: f[k].values[None, :] for k in f}
    p = {k: draws(post, k) for k in ("mu_ideal", "mu_beta", "mu_gamma", "mu_delta", "kappa",
                                     "rho", "omega", "eta", "chi", "over_frac", "mu_lapse")}
    theta, b = sigmoid(p["mu_ideal"]), np.exp(p["mu_beta"])

    def pen(rate):
        dev = rate - theta
        return dev ** 2 * np.where(dev > 0, p["over_frac"], 1.0)

    d = lambda k: v[f"{k}_a"] - v[f"{k}_b"]  # noqa: E731
    terms = {
        "switch rate vs ideal": -b * (pen(v["alt_rate_a"]) - pen(v["alt_rate_b"])),
        "heads/tails imbalance": -p["mu_gamma"] * d("imbalance"),
        "longest streak": -p["mu_delta"] * d("maxrun"),
        "final streak": -p["kappa"] * d("endrun"),
        "palindrome": -p["rho"] * d("palin"),
        "run-length variety": p["omega"] * d("variety"),
        "most lopsided stretch": -p["eta"] * d("lopsided"),
        "complement symmetry": -p["chi"] * d("antisym"),
    }
    lapse = sigmoid(p["mu_lapse"])
    return terms, lambda logit: 0.5 * lapse + (1 - lapse) * sigmoid(logit)


def run2_terms(mod, post, stim):
    f = pd.DataFrame([mod.compute_features(a, b) for a, b in zip(stim.a, stim.b)])
    v = {k: f[k].values[None, :] for k in f}
    p = {k: draws(post, k) for k in ("mu_ideal", "mu_log_beta", "lam", "mu_sw", "mu_kappa",
                                     "gamma", "rho", "tau", "tau_open")}
    theta = sigmoid(p["mu_ideal"])
    sens = np.exp(p["mu_log_beta"]) * np.exp(p["lam"] * v["log_len"])
    sw = sigmoid(p["mu_sw"])
    terms = {
        "tally span vs expected": sens * ((v["span_b"] - theta) ** 2 - (v["span_a"] - theta) ** 2),
        "switch rate vs ideal": 4.0 * p["mu_kappa"] * ((v["sw_b"] - sw) ** 2 - (v["sw_a"] - sw) ** 2),
        "three-flip chunk variety": p["gamma"] * v["trip_diff"],
        "rule-built": -p["rho"] * v["rule_diff"],
        "closing streak": -p["tau"] * v["term_diff"],
        "opening streak": -p["tau_open"] * v["init_diff"],
    }
    return terms, sigmoid


def run3_terms(mod, post, stim):
    rows = [{"sequence_a": a, "sequence_b": b, "participant_id": 0, "chose_left": 0}
            for a, b in zip(stim.a, stim.b)]
    o = mod.prepare_observed(rows)
    p = {k: draws(post, k) for k in ("eps", "kappa", "mu_a", "mu_b", "gamma", "c_pad", "mu_w",
                                     "d_switch2", "g_run", "mu_side")}
    from scipy.special import gammaln, logsumexp

    h, n = o["seq_heads"][None, :], o["seq_len"][None, :]
    k2 = 0.5 * p["kappa"]
    lbias = (gammaln(h + k2) + gammaln(n - h + k2) - gammaln(n + p["kappa"])
             - 2 * gammaln(k2) + gammaln(p["kappa"]))
    motif = []
    for per in mod.PERIODS:
        free = np.minimum(float(per), n)
        cmp_, mis = o[f"seq_cmp{per}"][None, :], o[f"seq_mis{per}"][None, :]
        motif.append(-free * np.log(2.0) + (cmp_ - mis) * np.log1p(-p["eps"]) + mis * np.log(p["eps"]))
    lmotif = logsumexp(np.stack(motif), axis=0) - np.log(len(mod.PERIODS))
    lmark = np.broadcast_to(o["seq_lmark"][None, :], lmotif.shape)
    lreg = logsumexp(np.stack([lmark, lbias, lmotif + p["mu_a"]]), axis=0)
    ia, ib = o["idx_a"], o["idx_b"]
    ls = np.exp(-p["gamma"] * np.log((n[:, ia] + p["c_pad"]) / (5.0 + p["c_pad"])))
    beta = p["mu_b"]
    terms = {
        "believed switch rate of chance": p["mu_w"] * o["switch_diff"][None, :] * ls,
        "chance rarely switches twice": beta * 0.5 * p["d_switch2"] * o["alt2_diff"][None, :] * ls,
        "gambler's fallacy (streaks due to end)": beta * 0.5 * p["g_run"] * o["gam_diff"][None, :] * ls,
        "explained by a regular process": -beta * (lreg[:, ia] - lreg[:, ib]) * ls,
    }
    # Which regular process explains each sequence best (posterior-mean log evidence).
    names = np.array(["switch/repeat-biased coin", "trick coin", "repeating motif"])
    best = np.stack([lmark.mean(0), lbias.mean(0), (lmotif + p["mu_a"]).mean(0)]).argmax(0)
    extra = {"regular_a": names[best[ia]], "regular_b": names[best[ib]]}
    side = p["mu_side"]
    return terms, (lambda logit: sigmoid(logit + side)), extra


BUILD = {1: run1_terms, 2: run2_terms, 3: run3_terms}


def main():
    for run in (1, 2, 3):
        name = ph.BEST_AGENT_MODEL[run]
        responses_path = SERIES / f"collected/run{run}/subjective_randomness/experiment3/model_loop/responses.csv"
        models_dir, seed, _ = ph.fit_source(name, run)
        fitted = fit_model(name, models_dir, responses_path, cache_dir=ph.ANALYSIS_ROOT / "mcmc_cache",
                           random_seed=seed, **ph.SAMPLER)
        post = fitted.idata.posterior
        mod = load(f"winner_{run}", models_dir / f"{name}.py")
        stim = stimuli(pd.read_csv(responses_path))
        built = BUILD[run](mod, post, stim)
        terms, link = built[0], built[1]
        extra = built[2] if len(built) > 2 else {}
        total = sum(terms.values())
        p_full = link(total).mean(0)
        r_full = np.corrcoef(p_full, stim.people)[0, 1]
        rows = []
        for t, val in terms.items():
            p_without = link(total - val).mean(0)
            rows.append({
                "term": t,
                "sd_logit": float(val.mean(0).std()),
                "share_decided": float(np.mean(np.sign(val.mean(0)) == np.sign(total.mean(0))) if t else 0),
                "r_full": r_full,
                "r_without": float(np.corrcoef(p_without, stim.people)[0, 1]),
            })
        table = pd.DataFrame(rows).sort_values("sd_logit", ascending=False)
        table.to_csv(OUT / f"winner_cues_run{run}.csv", index=False)
        per_stim = stim.assign(model=p_full, **{t: v.mean(0) for t, v in terms.items()}, **extra)
        per_stim.to_csv(OUT / f"winner_cue_terms_run{run}.csv", index=False)
        print(f"\n=== run {run}: {name} ({len(stim)} pairs; typical-participant r = {r_full:.3f})")
        print(table.drop(columns=["r_full", "share_decided"]).round(3).to_string(index=False))
        del fitted
        clear_fit_cache()


if __name__ == "__main__":
    main()

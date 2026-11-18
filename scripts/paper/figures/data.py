"""Figure 2's views of the figure data: the number game's pooled models on its held-out responses
(``number_game_unseen_view``) and each domain's rows, the runs collapsed (``forest_rows``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

NG_TEST_SETS = {
    "unseen_items": "the (example set, target) items no run drew",
    "unseen_example_sets": "the example sets no run drew",
}
# Subjective randomness and the number game score each run on its own data, so Figure 2 draws them from
# all runs pooled (the figure data's ``pooled``, whose labels of the runs' models are "<run>__<name>");
# reference games score every chain on one test (the pre-registered experiment), so its runs are drawn
# together.
POOLED_VIEW = {"subjective_randomness", "number_game"}
POOLED_LABEL_SEP = "__"  # sr_figure_data.POOLED_LABEL_SEP


def number_game_unseen_view(data: Dict[str, Any], unseen: Dict[str, Any], test_set: str) -> Dict[str, Any]:
    """Figure 2's number game: the pooled models scored on ``test_set`` of the unseen-items record, each
    against that test set's best seed. ``data`` is not changed."""
    pooled = data["pooled"]
    if unseen["winner_runs"] != pooled["winner_runs"]:
        raise RuntimeError(f"number game: the unseen-items record scores the winners {unseen['winner_runs']}, "
                           f"the figure data's are {pooled['winner_runs']}")
    scored = unseen[test_set]
    view = {**pooled, "best_seed": scored["best_seed"],
            "elpd": {m: r["lpd"] for m, r in scored["models"].items()},
            "vs_best_seed": {m: r["vs_best_seed"] for m, r in scored["models"].items()},
            **{key: {label: scored["strip"][label] for label in pooled[key]}
               for key in ("pruned_during_run", "others_at_end")}}
    return {**data, "pooled": view,
            "scoring": (f"log predictive density on {NG_TEST_SETS[test_set]} "
                        f"({scored['n_responses']:,} responses), the models fitted on all runs pooled")}


def _lookup(table: Dict[str, Any], domain: str, model: str, file: str) -> Any:
    if model not in table:
        raise KeyError(f"{domain}: {model!r} is not in {file}")
    return table[model]


def forest_rows(domain: str, data: Dict[str, Any], names: Dict[str, str],
                papers: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    """Figure 2's rows for one domain: the best discovered models (best first), the non-winning models
    (those live at the end and those pruned during the runs) as one strip, then the starting models (best
    first, each with the ``paper`` it implements or ``None``), each against the best starting model.

    A labelled row has ``diff`` and ``se`` (``None``: no paired SE recorded), a winner its ``run`` (its
    index in the domain's run order). The strip has each model's ``values``, ``models`` (its name in its
    run), ``runs`` and ``progress`` through its own run, 0 for the first it admitted to 1 for the last
    (``pruned_order`` and ``others_order``), so the collapsed runs overlay."""
    if domain in POOLED_VIEW:
        pooled = data["pooled"]
        run_names = pooled["runs"]
        winners = {m: pooled["vs_best_seed"][m] for m in pooled["winner_runs"]}
        winner_run = {m: run_names.index(run) for m, run in pooled["winner_runs"].items()}
        seeds = {m: pooled["vs_best_seed"][m] for m in pooled["elpd"] if m not in winners}
        every = {**pooled["pruned_during_run"], **pooled["others_at_end"]}
        order = {**pooled["pruned_order"], **pooled["others_order"]}
        runs = [{label.split(POOLED_LABEL_SEP, 1)[1]: v for label, v in every.items()
                 if label.startswith(f"{run}{POOLED_LABEL_SEP}")} for run in run_names]
        orders = [{label.split(POOLED_LABEL_SEP, 1)[1]: i for label, i in order.items()
                   if label.startswith(f"{run}{POOLED_LABEL_SEP}")} for run in run_names]
    else:
        if len({r["best_seed"] for r in data["runs"]}) != 1:
            raise RuntimeError(f"{domain}: the runs are scored against different best starting models")
        winners = {r["best_model"]: r["vs_best_seed"][r["best_model"]] for r in data["runs"]}
        winner_run = {r["best_model"]: k for k, r in enumerate(data["runs"])}
        seeds: Dict[str, Dict[str, Any]] = {}
        for r in data["runs"]:
            for m, v in r["vs_best_seed"].items():
                if m == r["best_model"]:
                    continue
                if m in seeds and abs(seeds[m]["diff"] - v["diff"]) > 1e-6:
                    raise RuntimeError(f"{domain}: {m!r} differs between runs on one test")
                seeds[m] = v
        runs = [{**r["pruned_during_run"], **r["others_at_end"]} for r in data["runs"]]
        orders = [{**r["pruned_order"], **r["others_order"]} for r in data["runs"]]
    strip: Dict[str, List[Any]] = {"values": [], "models": [], "runs": [], "progress": []}
    for k, (models, order) in enumerate(zip(runs, orders)):
        ranked = sorted(models, key=lambda m: _lookup(order, domain, m, "the admission orders"))
        strip["values"] += [models[m] for m in ranked]
        strip["models"] += ranked
        strip["runs"] += [k] * len(ranked)
        strip["progress"] += [i / (len(ranked) - 1) if len(ranked) > 1 else 0.5 for i in range(len(ranked))]

    def rows(kind, values, **extra):
        out = [{"kind": kind, "model": m, "label": _lookup(names, domain, m, "model_names.yaml"),
                "diff": v["diff"], "se": v["se"], **{k: f(m) for k, f in extra.items()}} for m, v in values.items()]
        return sorted(out, key=lambda r: -r["diff"])

    return [*rows("winner", winners, run=winner_run.get), {"kind": "nonwinning", "label": "Non-winning models", **strip},
            *rows("seed", seeds, paper=lambda m: _lookup(papers, domain, m, "seed_papers.yaml"))]

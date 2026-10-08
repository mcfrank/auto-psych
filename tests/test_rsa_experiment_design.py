"""Per-participant trial lists of the reference-game experiment (src.rsa.experiment.design)."""

import copy
from collections import Counter

import pytest

from src.rsa.design_space import context_pool
from src.rsa.experiment.design import (
    DOMAINS,
    EXPERIMENT_ASSETS_DIR,
    IMAGES_DIR,
    Design,
    TrialSpec,
    all_images,
    check_trial,
    trial_list,
    trial_lists,
)

DEMO = Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json")


def _tests(lst):
    return [t for t in lst["trials"] if t["phase"] == "test"]


def test_a_list_is_a_pure_function_of_design_seed_and_index():
    a = trial_list(DEMO, seed=3, list_index=5, n_catch=2)
    assert a == trial_list(DEMO, seed=3, list_index=5, n_catch=2)
    assert a != trial_list(DEMO, seed=3, list_index=6, n_catch=2)
    assert a != trial_list(DEMO, seed=4, list_index=5, n_catch=2)
    doc = trial_lists(DEMO, seed=3, n_lists=7, n_catch=2)
    assert doc["lists"][5] == a
    assert doc["design_sha256"] == DEMO.sha256


def test_every_design_trial_appears_once_plus_the_catch_trials():
    for k in range(20):
        tests = _tests(trial_list(DEMO, seed=0, list_index=k, n_catch=3))
        assert [t["trial_number"] for t in tests] == list(range(len(DEMO.specs) + 3))
        assert sorted(t["spec_index"] for t in tests if not t["is_catch"]) == list(range(len(DEMO.specs)))
        assert sum(t["condition"] == "catch" for t in tests) == 3
        for t in tests:
            if not t["is_catch"]:
                spec = DEMO.specs[t["spec_index"]]
                assert t["condition"] == spec.label
                assert t["objects"] == [list(r) for r in spec.objects]
                assert t["utterance"] == spec.utterance


def test_domains_have_enough_features_and_the_word_mapping_is_a_bijection():
    four = {name for name, d in DOMAINS.items() if len(d.features) >= 4}
    assert four == {"friend", "snowman", "sundae"}
    for k in range(50):
        for t in trial_list(DEMO, seed=1, list_index=k, n_catch=2)["trials"]:
            n_feat = len(t["objects"][0])
            words = [f.word for f in DOMAINS[t["item"]].features]
            assert len(words) >= n_feat
            if n_feat == 4:
                assert t["item"] in four
            # injective map of the columns into the domain's features
            assert len(t["feature_names"]) == n_feat == len(set(t["feature_names"]))
            assert set(t["feature_names"]) <= set(words)


def test_the_word_is_true_of_some_object_and_catch_trials_are_unambiguous():
    n_catch = 0
    for k in range(50):
        for t in trial_list(DEMO, seed=2, list_index=k, n_catch=2)["trials"]:
            if t["utterance"] is None:
                assert t["query"] == "prior" and t["word"] is None
                continue
            true_of = [i for i, row in enumerate(t["objects"]) if row[t["utterance"]]]
            assert true_of
            assert t["word"] == t["feature_names"][t["utterance"]]
            if t["condition"] in ("catch", "practice"):
                assert true_of == [t["catch_target"]]
                n_catch += t["condition"] == "catch"
    assert n_catch == 100


def test_catch_trials_take_the_shape_of_design_trials():
    shapes = {(len(s.objects), len(s.objects[0])) for s in DEMO.specs}
    for k in range(30):
        for t in _tests(trial_list(DEMO, seed=5, list_index=k, n_catch=4)):
            if t["is_catch"]:
                assert (len(t["objects"]), len(t["objects"][0])) in shapes


def test_screen_order_bases_and_images_follow_the_objects():
    for k in range(30):
        for t in trial_list(DEMO, seed=6, list_index=k, n_catch=2)["trials"]:
            n_obj = len(t["objects"])
            assert sorted(t["display_order"]) == list(range(n_obj))
            if n_obj <= 3:
                assert len(set(t["bases"])) == n_obj
            assert [cell["object"] for cell in t["screen"]] == t["display_order"]
            for cell in t["screen"]:
                assert cell["alt"].startswith(f"a {t['item']} with ")
                assert all((IMAGES_DIR / f).is_file() for f in cell["images"])


def test_screen_positions_and_words_vary_across_participants():
    first = Counter()
    words = Counter()
    for k in range(200):
        t = next(t for t in _tests(trial_list(DEMO, seed=7, list_index=k, n_catch=0)) if t["spec_index"] == 0)
        first[t["display_order"][0]] += 1
        words[(t["item"], t["word"])] += 1
    assert set(first) == {0, 1, 2}
    assert len(words) > 10


def test_domains_rotate_across_a_participants_trials():
    specs = [TrialSpec(objects=((0, 0), (0, 1), (1, 1)), utterance=1, label=f"t{i}") for i in range(12)]
    design = Design(name="rotation", specs=tuple(specs))
    for k in range(10):
        lst = trial_list(design, seed=8, list_index=k, n_catch=0)
        items = [t["item"] for t in lst["trials"]]
        assert all(a != b for a, b in zip(items, items[1:]))
        counts = Counter(items[1:])
        assert set(counts) == set(DOMAINS) and max(counts.values()) - min(counts.values()) <= 1


def test_every_game_of_the_design_space_can_be_drawn():
    pool = context_pool([(2, 2), (3, 3), (4, 4)], include_prior_queries=True)
    specs = tuple(TrialSpec.from_context(ctx, label=f"pool{i}") for i, ctx in enumerate(pool))
    design = Design(name="pool", specs=specs)
    lst = trial_list(design, seed=9, list_index=0, n_catch=5)
    assert len(_tests(lst)) == len(pool) + 5


def test_n_trials_draws_a_subset_per_list():
    lists = trial_lists(DEMO, seed=10, n_lists=10, n_catch=1, n_trials=4)["lists"]
    drawn = [sorted(t["spec_index"] for t in _tests(lst) if not t["is_catch"]) for lst in lists]
    assert all(len(d) == 4 == len(set(d)) for d in drawn)
    assert len({tuple(d) for d in drawn}) > 1


def test_message_sets_are_carried():
    spec = TrialSpec(objects=((1, 0, 1), (0, 1, 1)), utterance=0, label="m", messages=(0, 1))
    lst = trial_list(Design(name="m", specs=(spec,)), seed=0, list_index=0, n_catch=0)
    assert _tests(lst)[0]["messages"] == [0, 1]


def test_invalid_designs_raise():
    with pytest.raises(ValueError, match="true of no object"):
        TrialSpec(objects=((0, 0), (0, 1)), utterance=0, label="x")
    with pytest.raises(ValueError, match="reserved"):
        Design(name="d", specs=(TrialSpec(objects=((0, 1), (1, 1)), utterance=1, label="catch"),))
    with pytest.raises(ValueError, match="no domain has that many"):
        Design(name="d", specs=(TrialSpec(objects=((1,) * 6, (0,) * 5 + (1,)), utterance=0, label="wide"),))
    with pytest.raises(ValueError, match="roles"):
        TrialSpec(objects=((0, 1), (1, 1)), utterance=1, label="x", roles=("a",))
    with pytest.raises(ValueError, match="unknown trial spec keys"):
        Design.from_json({"name": "d", "trials": [{"objects": [[1]], "utterance": 0, "label": "x", "word": "hat"}]})


def test_a_tampered_trial_fails_the_check():
    trial = _tests(trial_list(DEMO, seed=11, list_index=0, n_catch=2))[0]
    check_trial(trial)
    bad = copy.deepcopy(trial)
    bad["feature_names"][0] = bad["feature_names"][-1] if len(bad["feature_names"]) > 1 else "nonsense"
    with pytest.raises(ValueError):
        check_trial(bad)
    bad = copy.deepcopy(trial)
    bad["display_order"] = bad["display_order"][::-1]
    with pytest.raises(ValueError, match="screen position"):
        check_trial(bad)


def test_every_image_a_domain_can_use_is_in_the_project_assets():
    files = all_images()
    assert len(files) == len(set(files)) == 40
    assert all((IMAGES_DIR / f).is_file() for f in files)
    assert sorted(p.name for p in IMAGES_DIR.glob("*.png")) == sorted(files)


def test_subsets_are_balanced_across_consecutive_lists():
    # With 12 trials a person sees part of the design: every display must get
    # the same number of people, not the luck of independent random subsets.
    specs = tuple(TrialSpec(objects=((0, 0), (0, 1), (1, 1)), utterance=1, label=f"d{i}") for i in range(30))
    design = Design(name="subsets", specs=specs)
    lists = trial_lists(design, seed=11, n_lists=5, n_catch=2, n_trials=12)["lists"]  # 5 x 12 = 2 x 30
    seen = Counter(t["spec_index"] for lst in lists for t in _tests(lst) if not t["is_catch"])
    assert set(seen) == set(range(30)) and set(seen.values()) == {2}
    for lst in lists:
        drawn = [t["spec_index"] for t in _tests(lst) if not t["is_catch"]]
        assert len(drawn) == 12 == len(set(drawn))

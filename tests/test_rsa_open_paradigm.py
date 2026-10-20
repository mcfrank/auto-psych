"""The paradigm spec the designer writes, and its validator (docs/auto_open/PLAN.md §3)."""

import copy
import json

import pytest

from src.rsa.loop.novelty import plain_pool

SQUARE = '<rect x="10" y="10" width="20" height="20" fill="#2a78d6"/>'


def closed_like_spec():
    """The closed campaign's paradigm (forced choice over the 1,597-display pool, 200 people,
    13 + 2 trials, $0.50 for 3 minutes), with stand-in SVG materials."""
    return {
        "schema_version": 1,
        "name": "reference game, forced choice",
        "summary": "Participants see a few creatures, hear one word, and click the one they think the speaker means.",
        "charter": {"question": "How do listeners use the alternatives a speaker could have said to pick a referent?"},
        "instructions": {
            "welcome": "In this study you will play a short guessing game.",
            "task": "A speaker describes one of the creatures with one word. Click the creature you think the speaker means.",
            "comprehension_checks": [
                {"question": "What do you do on each screen?",
                 "options": ["Click the creature the speaker means", "Type a word"], "correct": 0},
            ],
        },
        "materials": {
            "kind": "svg",
            "domains": [{
                "name": "creatures",
                "base": '<circle cx="50" cy="50" r="40" fill="#e1e0d9"/>',
                "features": [
                    {"word": "hat", "svg": SQUARE},
                    {"word": "scarf", "svg": SQUARE},
                    {"word": "glasses", "svg": SQUARE},
                    {"word": "boots", "svg": SQUARE},
                ],
            }],
        },
        "scene_space": {
            "sizes": [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4]],
            "prior_queries": True,
            "synonyms": True,
        },
        "response": {"type": "choice", "link": "choice",
                     "prompt": "Which creature is the speaker talking about?"},
        "design": {"n_participants": 200, "trials_per_participant": 13, "distinct_trials": 52, "catch_trials": 2,
                   "minutes": 3, "pay_usd": 0.50},
        "content": {"deception": False, "affective_stimuli": False},
    }


def write(tmp_path, spec, name="paradigm.json"):
    path = tmp_path / name
    path.write_text(json.dumps(spec, indent=2))
    return path


def test_the_closed_paradigm_as_a_spec_validates_and_gives_the_closed_design_pool(tmp_path):
    from src.rsa_open.paradigm import load_paradigm, scene_pool

    paradigm = load_paradigm(write(tmp_path, closed_like_spec()))
    assert paradigm.cost_usd == pytest.approx(133.33, abs=0.01)
    assert scene_pool(paradigm) == plain_pool()


def test_every_problem_is_reported_at_once(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["design"].update(n_participants=50, minutes=20)
    spec["materials"]["domains"][0]["features"][0]["svg"] = "<script>alert(1)</script>"
    with pytest.raises(ParadigmInvalid) as err:
        load_paradigm(write(tmp_path, spec))
    message = str(err.value)
    assert "n_participants" in message and "minutes" in message and "script" in message


@pytest.mark.parametrize("svg, problem", [
    ("<script>x</script>", "script"),
    ('<rect onclick="x()" width="1" height="1"/>', "onclick"),
    ('<image href="https://example.com/a.png"/>', "image"),
    ('<use href="https://example.com/a.svg#x"/>', "href"),
    ("<foreignObject><div>hi</div></foreignObject>", "foreignObject"),
    ("<rect", "not well-formed"),
])
def test_unsafe_or_broken_svg_is_refused(tmp_path, svg, problem):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["materials"]["domains"][0]["features"][1]["svg"] = svg
    with pytest.raises(ParadigmInvalid, match=problem):
        load_paradigm(write(tmp_path, spec))


def test_the_budget_is_the_closed_campaign_s_and_pay_is_at_its_rate(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    over = closed_like_spec()
    over["design"].update(n_participants=150, minutes=4, pay_usd=0.80)  # $159.60
    with pytest.raises(ParadigmInvalid, match="budget"):
        load_paradigm(write(tmp_path, over))
    underpaid = closed_like_spec()
    underpaid["design"].update(minutes=6, pay_usd=0.50, n_participants=100)
    with pytest.raises(ParadigmInvalid, match="per hour"):
        load_paradigm(write(tmp_path, underpaid))
    longer = closed_like_spec()
    longer["design"].update(n_participants=100, minutes=6, pay_usd=1.00)  # same dollars, other trade-off
    assert load_paradigm(write(tmp_path, longer)).cost_usd == pytest.approx(133.33, abs=0.01)


def test_a_response_type_without_a_link_in_the_library_is_refused(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["response"].update(type="slider", link="slider")
    with pytest.raises(ParadigmInvalid, match="link"):
        load_paradigm(write(tmp_path, spec))


@pytest.mark.parametrize("field, text", [
    (("instructions", "task"), "Click the creature covered in blood."),
    (("response", "prompt"), "Which one would you kill first?"),
])
def test_charged_language_is_refused(tmp_path, field, text):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec[field[0]][field[1]] = text
    with pytest.raises(ParadigmInvalid, match="charged"):
        load_paradigm(write(tmp_path, spec))


def test_deception_or_affective_stimuli_are_refused(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["content"]["deception"] = True
    with pytest.raises(ParadigmInvalid, match="deception"):
        load_paradigm(write(tmp_path, spec))


def test_links_and_requests_for_personal_information_are_refused(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["instructions"]["welcome"] = "Please email us at someone@example.com or see https://example.com"
    with pytest.raises(ParadigmInvalid, match="URL|email"):
        load_paradigm(write(tmp_path, spec))


def test_each_domain_must_name_enough_features_for_the_largest_scene(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["materials"]["domains"][0]["features"] = spec["materials"]["domains"][0]["features"][:3]
    with pytest.raises(ParadigmInvalid, match="4 features"):
        load_paradigm(write(tmp_path, spec))


def test_a_scene_pool_beyond_the_bound_is_refused(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["scene_space"]["sizes"] = [[7, 2]]
    with pytest.raises(ParadigmInvalid, match="6 objects"):
        load_paradigm(write(tmp_path, spec))


def test_text_materials_need_a_phrase_per_feature(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["materials"] = {"kind": "text", "domains": [{
        "name": "houses", "base": "a house",
        "features": [{"word": w, "phrase": f"with a {w}"} for w in ("garden", "garage", "porch", "pool")]}]}
    assert load_paradigm(write(tmp_path, spec)).materials.kind == "text"
    spec["materials"]["domains"][0]["features"][0].pop("phrase")
    with pytest.raises(ParadigmInvalid, match="phrase"):
        load_paradigm(write(tmp_path, spec))


def test_the_paradigm_hash_ignores_formatting_but_not_content(tmp_path):
    from src.rsa_open.paradigm import load_paradigm

    spec = closed_like_spec()
    a = load_paradigm(write(tmp_path, spec, "a.json")).sha256
    reordered = json.loads(json.dumps(spec), object_pairs_hook=lambda kv: dict(reversed(kv)))
    (tmp_path / "b.json").write_text(json.dumps(reordered))
    assert load_paradigm(tmp_path / "b.json").sha256 == a
    changed = copy.deepcopy(spec)
    changed["response"]["prompt"] = "Which creature does the speaker mean?"
    assert load_paradigm(write(tmp_path, changed, "c.json")).sha256 != a


def test_the_schema_for_the_designer_lists_every_field(tmp_path):
    from src.rsa_open.paradigm import main_schema

    schema = main_schema(tmp_path / "schema.json")
    assert {"materials", "scene_space", "response", "design", "content"} <= set(schema["properties"])


@pytest.mark.parametrize("response, ok", [
    ({"type": "rating", "link": "rating", "prompt": "How likely is it the speaker means this one?",
      "scale_points": 7}, True),
    ({"type": "rating", "link": "rating", "prompt": "How likely?"}, False),
    ({"type": "slider", "link": "allocation", "prompt": "Split 100 points by how likely each is.",
      "total_points": 100}, True),
    ({"type": "slider", "link": "allocation", "prompt": "Split the points.", "total_points": 1}, False),
    ({"type": "choice", "link": "choice", "prompt": "Which one?", "scale_points": 7}, False),
    ({"type": "slider", "link": "rating", "prompt": "Which one?", "total_points": 100}, False),
])
def test_rating_and_slider_paradigms_fix_their_scale(tmp_path, response, ok):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm, paradigm_likelihood

    spec = closed_like_spec()
    spec["response"] = response
    if ok:
        link = paradigm_likelihood(load_paradigm(write(tmp_path, spec)))
        assert link.name == response["link"]
        assert link.options == ({"scale_points": 7} if link.name == "rating" else {"total_points": 100})
    else:
        with pytest.raises(ParadigmInvalid, match="points|scores"):
            load_paradigm(write(tmp_path, spec))


def test_the_design_says_how_many_distinct_trials_it_picks(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["design"]["distinct_trials"] = 10  # fewer than each participant answers
    with pytest.raises(ParadigmInvalid, match="distinct_trials"):
        load_paradigm(write(tmp_path, spec))
    spec["design"].pop("distinct_trials")
    with pytest.raises(ParadigmInvalid, match="distinct_trials"):
        load_paradigm(write(tmp_path, spec))

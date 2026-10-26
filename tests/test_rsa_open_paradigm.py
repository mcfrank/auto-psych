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
                    {"word": "hat", "svg": '<path d="M30 28 L70 28 L62 8 L38 8 Z" fill="#3b3b98"/>'},
                    {"word": "scarf", "svg": '<path d="M28 70 Q50 82 72 70 L72 77 Q50 89 28 77 Z" fill="#c0392b"/>'},
                    {"word": "glasses", "svg": '<g fill="none" stroke="#222222" stroke-width="3">'
                                               '<circle cx="40" cy="48" r="7"/><circle cx="60" cy="48" r="7"/></g>'},
                    {"word": "boots", "svg": '<rect x="34" y="88" width="12" height="8" fill="#5d4037"/>'
                                             '<rect x="54" y="88" width="12" height="8" fill="#5d4037"/>'},
                ],
            }],
        },
        "scene_space": {
            "sizes": [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4]],
            "prior_queries": True,
            "synonyms": True,
        },
        "response": response("choice"),
        "design": {"n_participants": 200, "trials_per_participant": 13, "distinct_trials": 52, "catch_trials": 2,
                   "minutes": 3, "pay_usd": 0.50},
        "content": {"deception": False, "affective_stimuli": False},
    }


LISTENER_PROMPT = "The speaker said {message}. Which creature is the speaker talking about?"
PRIOR_PROMPT = "The speaker said a word you could not hear. Which creature is the speaker talking about?"
LINK_OF = {"choice": "choice", "rating": "rating", "slider": "ordered_beta", "bet": "allocation",
           "production": "production", "binary_judgment": "binary_judgment"}


def response(kind, **extra):
    """A valid response block of each type, for a scene space with prior queries."""
    out = {"type": kind, "link": LINK_OF[kind]}
    if kind in ("choice", "rating", "slider", "bet"):
        out.update(prompt=LISTENER_PROMPT, prior_prompt=PRIOR_PROMPT)
    elif kind == "production":
        out["prompt"] = "Which word would you use to point out the marked creature?"
    else:
        out["prompt"] = "Would a speaker say {message} to point out the marked creature?"
    if kind == "rating":
        out.update(scale_points=5, scale_labels=["not at all", "certainly"])
    if kind == "slider":
        out.update(total_points=100, scale_labels=["not at all", "certainly"])
    if kind == "bet":
        out["total_points"] = 100
    return {**out, **extra}


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


def test_the_budget_is_twice_the_closed_campaign_s_and_pay_is_at_its_rate(tmp_path):
    from src.rsa_open.paradigm import BUDGET_USD, ParadigmInvalid, load_paradigm

    assert BUDGET_USD == pytest.approx(266.67, abs=0.01)  # 2 x (200 people x $0.50 x Prolific's 4/3)
    at_budget = closed_like_spec()
    at_budget["design"].update(n_participants=400)  # the closed paradigm at twice its people
    assert load_paradigm(write(tmp_path, at_budget)).cost_usd == pytest.approx(266.67, abs=0.01)
    over = closed_like_spec()
    over["design"].update(n_participants=300, minutes=4, pay_usd=0.80)  # $320
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
    (("response", "prompt"), "The speaker said {message}. Which one would you kill first?"),
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
    changed["response"]["prompt"] = "The speaker said {message}. Which creature does the speaker mean?"
    assert load_paradigm(write(tmp_path, changed, "c.json")).sha256 != a


def test_the_schema_for_the_designer_lists_every_field(tmp_path):
    from src.rsa_open.paradigm import main_schema

    schema = main_schema(tmp_path / "schema.json")
    assert {"materials", "scene_space", "response", "design", "content"} <= set(schema["properties"])


@pytest.mark.parametrize("block, ok", [
    (response("rating", scale_points=7), True),
    (response("rating", scale_points=None), False),
    (response("bet"), True),
    (response("slider"), True),
    (response("slider", total_points=1), False),
    (response("slider", link="allocation"), False),
    (response("choice", scale_points=7), False),
    (response("slider", link="rating"), False),
], ids=lambda x: str(x) if isinstance(x, bool) else f"{x['type']}-{x['link']}")
def test_rating_and_slider_paradigms_fix_their_scale(tmp_path, block, ok):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm, paradigm_likelihood

    spec = closed_like_spec()
    spec["response"] = block
    if ok:
        link = paradigm_likelihood(load_paradigm(write(tmp_path, spec)))
        assert link.name == block["link"]
        expected = {"rating": {"scale_points": block.get("scale_points")}}.get(link.name, {"total_points": 100})
        assert link.options == expected
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


@pytest.mark.parametrize("kind", ["choice", "rating", "slider", "bet", "production", "binary_judgment"])
def test_a_valid_response_block_of_each_type(tmp_path, kind):
    from src.rsa_open.paradigm import load_paradigm

    spec = closed_like_spec()
    spec["response"] = response(kind)
    if kind in ("production", "binary_judgment"):
        spec["response"].pop("prior_prompt", None)
    assert load_paradigm(write(tmp_path, spec)).response.type == kind


@pytest.mark.parametrize("change, problem", [
    ({"prompt": "Which creature is the speaker talking about?"}, "once"),
    ({"prompt": "The speaker said {message} or {message}."}, "once"),
    ({"prompt": "The speaker said {word}. Which one?"}, "unknown placeholders"),
    ({"prompt": 'The speaker says: "{message}". Which one?'}, "quotation marks"),
    ({"prompt": "The speaker says “{message}”. Which one?"}, "quotation marks"),
    ({"prior_prompt": None}, "prior_prompt is required"),
    ({"prior_prompt": "The speaker said {message}."}, "no message"),
])
def test_prompts_show_the_message_where_they_say_so(tmp_path, change, problem):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["response"].update(change)
    with pytest.raises(ParadigmInvalid, match=problem):
        load_paradigm(write(tmp_path, spec))


def test_a_production_prompt_has_no_message_and_needs_no_prior_prompt(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    spec = closed_like_spec()
    spec["response"] = response("production", prompt="The speaker said {message}.")
    with pytest.raises(ParadigmInvalid, match="no message to show"):
        load_paradigm(write(tmp_path, spec))
    spec["response"] = response("production", prior_prompt="Which one?")
    with pytest.raises(ParadigmInvalid, match="prior_prompt applies only"):
        load_paradigm(write(tmp_path, spec))


def test_ratings_and_sliders_label_their_ends(tmp_path):
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    for kind in ("rating", "slider"):
        spec = closed_like_spec()
        spec["response"] = response(kind, scale_labels=["low"])
        with pytest.raises(ParadigmInvalid, match="scale_labels"):
            load_paradigm(write(tmp_path, spec))
    spec = closed_like_spec()
    spec["response"]["scale_labels"] = ["a", "b"]
    with pytest.raises(ParadigmInvalid, match="scale_labels applies only"):
        load_paradigm(write(tmp_path, spec))

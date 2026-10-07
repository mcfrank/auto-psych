"""Rendering the reference-game page (src.rsa.experiment.build)."""

import json
import re

import pytest

from src.rsa.experiment.build import (
    CONSENT_PLACEHOLDER_MARKER,
    Args,
    build_single_file,
    build_site,
    lists_from_args,
    render_page,
    used_images,
)
from src.rsa.experiment.design import EXPERIMENT_ASSETS_DIR, Design, trial_lists

DEMO_PATH = EXPERIMENT_ASSETS_DIR / "demo_design.json"


def _doc(n_lists=3):
    return trial_lists(Design.load(DEMO_PATH), seed=31, n_lists=n_lists, n_catch=2)


def _embedded(html, name):
    m = re.search(rf"const {name} = (.*?);\n", html)
    assert m, name
    return json.loads(m.group(1))


def test_the_site_holds_the_page_the_used_images_and_the_lists(tmp_path):
    doc = _doc()
    index = build_site(doc, tmp_path / "site")
    html = index.read_text(encoding="utf-8")
    assert "{{" not in html
    assert "jsPsychHtmlButtonResponse" in html and "https://unpkg.com/jspsych@8.2.3" in html
    assert "window.submitData" in html
    assert CONSENT_PLACEHOLDER_MARKER in html
    assert _embedded(html, "TRIAL_LISTS")["lists"] == doc["lists"]
    assert _embedded(html, "PREVIEW") is False
    urls = _embedded(html, "IMAGE_URLS")
    assert urls == {f: f"images/{f}" for f in used_images(doc)}
    assert sorted(p.name for p in (tmp_path / "site" / "images").iterdir()) == used_images(doc)
    assert json.loads((tmp_path / "site" / "trial_lists.json").read_text()) == doc
    # The only remote resources are jsPsych's, from unpkg.
    remote = set(re.findall(r"https?://[^\s\"')]+", html))
    assert all(u.startswith("https://unpkg.com/") for u in remote), remote


def test_a_non_empty_site_dir_is_not_overwritten_silently(tmp_path):
    doc = _doc(1)
    build_site(doc, tmp_path)
    with pytest.raises(FileExistsError):
        build_site(doc, tmp_path)
    build_site(doc, tmp_path, overwrite=True)


def test_the_single_file_inlines_every_image(tmp_path):
    doc = _doc(1)
    path = build_single_file(doc, tmp_path / "preview.html", preview=True)
    html = path.read_text(encoding="utf-8")
    urls = _embedded(html, "IMAGE_URLS")
    assert set(urls) == set(used_images(doc))
    assert all(u.startswith("data:image/png;base64,") for u in urls.values())
    assert _embedded(html, "PREVIEW") is True


def test_embedded_text_cannot_close_the_script():
    doc = _doc(1)
    html = render_page(doc, {f: f for f in used_images(doc)}, preview=False, consent_html="<p>ok</p></script><b>x</b>")
    assert html.count("</script>") == html.count("<script")
    assert "<\\/script>" in html


def test_a_missing_image_raises(tmp_path):
    doc = _doc(1)
    with pytest.raises(ValueError, match="no URL for images"):
        render_page(doc, {}, preview=False, consent_html="")
    with pytest.raises(FileNotFoundError):
        build_site(doc, tmp_path / "s", images_dir=tmp_path)


def test_the_cli_takes_lists_or_a_design_but_not_both(tmp_path):
    lists_path = tmp_path / "lists.json"
    lists_path.write_text(json.dumps(_doc(2)))
    assert len(lists_from_args(Args(trial_lists=lists_path))["lists"]) == 2
    assert len(lists_from_args(Args(design=DEMO_PATH, seed=1, n_lists=3))["lists"]) == 3
    with pytest.raises(ValueError, match="exactly one"):
        lists_from_args(Args(trial_lists=lists_path, design=DEMO_PATH))
    with pytest.raises(ValueError, match="needs --seed"):
        lists_from_args(Args(design=DEMO_PATH))
    with pytest.raises(ValueError, match="only with --design"):
        lists_from_args(Args(trial_lists=lists_path, seed=3))

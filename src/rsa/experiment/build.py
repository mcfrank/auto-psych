"""Render the reference-game experiment into a static site or a single HTML file.

    uv run python -m src.rsa.experiment.build --trial-lists lists.json --out-dir site/
    uv run python -m src.rsa.experiment.build --design design.json --seed 1 --n-lists 40 \\
        --out-dir site/ --single-file preview.html --preview

The site is ``index.html`` (the jsPsych page; jsPsych itself loads from the
unpkg CDN, nothing else is fetched), ``images/`` (only the images the lists
use) and ``trial_lists.json`` (the full lists document, for the record). The
single file has the images inlined as ``data:`` URIs, for a preview that
opens from disk. ``--preview`` makes the end screen show the recorded data.

The page runs list ``k`` with ``?list=k`` and a random list otherwise
(`src.rsa.experiment.design`). Without ``--consent-html`` the consent screen
is a marked placeholder (`CONSENT_PLACEHOLDER_MARKER`), which a live
deployment must replace.
"""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tyro

from src.rsa.experiment.design import IMAGES_DIR, Design, load_trial_lists, trial_lists

TEMPLATE = Path(__file__).with_name("template.html")
CONSENT_PLACEHOLDER_MARKER = "rsa-placeholder"
CONSENT_PLACEHOLDER = (
    '<div class="rsa-placeholder"><h1>Consent</h1>'
    "<p><strong>[CONSENT PLACEHOLDER]</strong> The approved consent text goes here. "
    "This build is a prototype and must not be shown to real participants.</p>"
    "<p>Click “I agree” to continue.</p></div>"
)
PLACEHOLDERS = ("{{TRIAL_LISTS_JSON}}", "{{IMAGE_URLS_JSON}}", "{{PREVIEW_JSON}}", "{{CONSENT_HTML_JSON}}")


def _script_json(value) -> str:
    """JSON safe to embed in a <script> element."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")


def used_images(lists_doc: dict) -> List[str]:
    files = {f for lst in lists_doc["lists"] for t in lst["trials"] for cell in t["screen"] for f in cell["images"]}
    return sorted(files)


def render_page(lists_doc: dict, image_urls: Dict[str, str], *, preview: bool, consent_html: str) -> str:
    missing = sorted(set(used_images(lists_doc)) - set(image_urls))
    if missing:
        raise ValueError(f"no URL for images {missing}")
    page_lists = {
        "design_name": lists_doc["design_name"],
        "design_sha256": lists_doc["design_sha256"],
        "lists": lists_doc["lists"],
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for placeholder in PLACEHOLDERS:
        if html.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
    html = (
        html.replace("{{TRIAL_LISTS_JSON}}", _script_json(page_lists))
        .replace("{{IMAGE_URLS_JSON}}", _script_json(image_urls))
        .replace("{{PREVIEW_JSON}}", _script_json(bool(preview)))
        .replace("{{CONSENT_HTML_JSON}}", _script_json(consent_html))
    )
    return html


def _image_path(images_dir: Path, file: str) -> Path:
    path = Path(images_dir) / file
    if not path.is_file():
        raise FileNotFoundError(f"image {file} not found in {images_dir}")
    return path


def build_site(
    lists_doc: dict,
    out_dir: Path,
    *,
    preview: bool = False,
    consent_html: str = CONSENT_PLACEHOLDER,
    images_dir: Path = IMAGES_DIR,
    overwrite: bool = False,
) -> Path:
    """Write index.html, images/ and trial_lists.json into ``out_dir``; return index.html."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty (pass overwrite=True / --overwrite)")
    files = used_images(lists_doc)
    sources = {f: _image_path(images_dir, f) for f in files}
    html = render_page(lists_doc, {f: f"images/{f}" for f in files}, preview=preview, consent_html=consent_html)
    if (out_dir / "images").exists():
        shutil.rmtree(out_dir / "images")
    (out_dir / "images").mkdir(parents=True)
    for f, src in sources.items():
        shutil.copyfile(src, out_dir / "images" / f)
    (out_dir / "trial_lists.json").write_text(json.dumps(lists_doc, indent=1) + "\n", encoding="utf-8")
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def build_single_file(
    lists_doc: dict,
    path: Path,
    *,
    preview: bool = False,
    consent_html: str = CONSENT_PLACEHOLDER,
    images_dir: Path = IMAGES_DIR,
) -> Path:
    """One HTML file with every image inlined as a data: URI."""
    urls = {
        f: "data:image/png;base64," + base64.b64encode(_image_path(images_dir, f).read_bytes()).decode("ascii")
        for f in used_images(lists_doc)
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_page(lists_doc, urls, preview=preview, consent_html=consent_html), encoding="utf-8")
    return path


@dataclass
class Args:
    """Render the jsPsych reference-game page from trial lists (or a design)."""

    trial_lists: Optional[Path] = None
    """Trial lists JSON written by `python -m src.rsa.experiment.design` (or give --design)."""
    design: Optional[Path] = None
    """Design JSON to generate lists from (with --seed and --n-lists)."""
    seed: Optional[int] = None
    """With --design: base seed of the lists."""
    n_lists: Optional[int] = None
    """With --design: number of participant lists to bake in."""
    n_catch: int = 2
    """With --design: catch trials per list."""
    n_trials: Optional[int] = None
    """With --design: design trials per list (default: all)."""
    out_dir: Optional[Path] = None
    """Static site directory to write (index.html, images/, trial_lists.json)."""
    single_file: Optional[Path] = None
    """Single HTML file with images inlined (a preview that opens from disk)."""
    preview: bool = False
    """Show the recorded data on the end screen."""
    consent_html: Optional[Path] = None
    """File with the consent screen's HTML (default: a marked placeholder)."""
    overwrite: bool = False
    """Allow writing into a non-empty --out-dir."""


def lists_from_args(args: Args) -> dict:
    if (args.trial_lists is None) == (args.design is None):
        raise ValueError("give exactly one of --trial-lists and --design")
    if args.trial_lists is not None:
        if args.seed is not None or args.n_lists is not None or args.n_trials is not None:
            raise ValueError("--seed/--n-lists/--n-trials apply only with --design")
        return load_trial_lists(args.trial_lists)
    if args.seed is None or args.n_lists is None:
        raise ValueError("--design needs --seed and --n-lists")
    return trial_lists(
        Design.load(args.design), seed=args.seed, n_lists=args.n_lists, n_catch=args.n_catch, n_trials=args.n_trials
    )


def main(args: Args) -> None:
    if args.out_dir is None and args.single_file is None:
        raise ValueError("give --out-dir and/or --single-file")
    lists_doc = lists_from_args(args)
    consent = CONSENT_PLACEHOLDER if args.consent_html is None else args.consent_html.read_text(encoding="utf-8")
    if args.out_dir is not None:
        index = build_site(lists_doc, args.out_dir, preview=args.preview, consent_html=consent, overwrite=args.overwrite)
        print(f"wrote {index} ({len(lists_doc['lists'])} lists, {len(used_images(lists_doc))} images)")
    if args.single_file is not None:
        path = build_single_file(lists_doc, args.single_file, preview=args.preview, consent_html=consent)
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(tyro.cli(Args))

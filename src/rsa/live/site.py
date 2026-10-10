"""The reference-game page as the live deployment serves it.

`build_live_site` writes ``<exp_dir>/experiment/`` (index.html, images/,
trial_lists.json), the layout main's deployment stages
(`src.pipelines.outer_loop.deployment.local.run_deployment`). The page:

* gets its list from ``POST /assign`` (`functions/index.js`): the next list of
  the collection session in arrival order, the same list again on a reload. So
  the 200 balanced lists are used once each, and every designed display gets
  its share of responses. Without a participant id the page stops: it never
  falls back to a random list;
* shows no consent screen of its own: the deployment injects the IRB consent
  gate (`templates/consent.txt`) in front of it;
* posts the jsPsych data to ``POST /submit`` with the client config the
  deployment writes beside it (``auto_psych_config.json``: the collection
  session, the study), the Prolific ids from the URL and the list, then sends
  the participant to Prolific's completion URL. A failed post shows an error
  and does not redirect. Because the page posts to ``/submit`` itself, the
  deployment does not inject its generic submit bridge.
"""

from __future__ import annotations

from pathlib import Path

from src.rsa.experiment.build import CONSENT_PLACEHOLDER_MARKER, build_site

LIVE_MARKER = "auto-psych-rsa-live"
_INSERT_BEFORE = "  <script>\n    // ---- Injected by `python -m src.rsa.experiment.build`"

LIVE_JS = """
// auto-psych-rsa-live: the list from /assign, the data to /submit (src/rsa/live/site.py).
(function () {
  var configPromise = null;
  function config() {
    if (!configPromise) {
      configPromise = fetch("auto_psych_config.json", { cache: "no-store" }).then(function (r) {
        if (!r.ok) throw new Error("the study configuration could not be loaded (" + r.status + ")");
        return r.json();
      });
    }
    return configPromise;
  }
  function params() { return new URLSearchParams(window.location.search); }
  function participantKey() { var p = params(); return p.get("PROLIFIC_PID") || p.get("participant_id"); }
  function failed(what) {
    return function (r) {
      if (r.ok) return r;
      return r.text().then(function (t) { throw new Error(what + " (" + r.status + "): " + t); });
    };
  }
  window.assignList = function (n) {
    var key = participantKey();
    if (!key) return Promise.reject(new Error("this link has no participant id"));
    return config().then(function (cfg) {
      return fetch("/assign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_session_id: cfg.collection_session_id, n_lists: n, participant_key: key })
      });
    }).then(failed("list assignment failed")).then(function (r) { return r.json(); }).then(function (d) {
      return d.list_index;
    });
  };
  window.submitData = function (json, meta) {
    var p = params();
    return config().then(function (cfg) {
      var payload = Object.assign({}, cfg, {
        participant_id: participantKey(),
        prolific_pid: p.get("PROLIFIC_PID"),
        prolific_study_id_from_url: p.get("STUDY_ID"),
        prolific_session_id: p.get("SESSION_ID"),
        list_index: meta.list_index,
        trials: JSON.parse(json),
        consented_at: window.__autoPsychConsentedAt || null,
        submitted_at_client: new Date().toISOString(),
        user_agent: navigator.userAgent
      });
      return fetch("/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(failed("your responses could not be saved")).then(function () {
        if (cfg.prolific_redirect_url) window.location.href = cfg.prolific_redirect_url;
      });
    });
  };
})();
""".strip()


def live_page(html: str) -> str:
    """The built page with the live hooks, defined before the page's own script."""
    if LIVE_MARKER in html:
        return html
    if html.count(_INSERT_BEFORE) != 1:
        raise ValueError("the page has no single injection point for the live hooks")
    i = html.index(_INSERT_BEFORE)
    return html[:i] + f'  <script id="{LIVE_MARKER}">\n{LIVE_JS}\n  </script>\n' + html[i:]


def build_live_site(lists_doc: dict, exp_dir: Path, *, overwrite: bool = False) -> Path:
    """``<exp_dir>/experiment/`` for main's deployment; returns its index.html."""
    site = Path(exp_dir) / "experiment"
    index = build_site(lists_doc, site, preview=False, consent_html=None, overwrite=overwrite)
    html = live_page(index.read_text(encoding="utf-8"))
    if "[CONSENT PLACEHOLDER]" in html or f'class="{CONSENT_PLACEHOLDER_MARKER}"' in html:
        raise ValueError("a live page must not carry the consent placeholder")
    index.write_text(html, encoding="utf-8")
    return index

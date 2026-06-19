"use strict";
/* auto-psych study monitor — a dependency-free dashboard for live human studies.
 *
 * It polls /api/sessions for the list of studies and /api/session/<id> for the
 * open study, redrawing on each tick. The emphasis throughout is data quality:
 * the degenerate-data state (everyone choosing one side) is surfaced loudly so a
 * broken study is caught within minutes, not at analysis time. */

const REFRESH_MS = 15000;

const state = {
  selected: null, // collection_session_id of the open study
  sessions: [],
  detail: null,
  timer: null,
  countdown: REFRESH_MS / 1000,
};

const $ = (id) => document.getElementById(id);

// ── fetch helpers ──────────────────────────────────────────────────────
async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

// ── formatting ─────────────────────────────────────────────────────────
function pct(x) {
  return x == null ? "—" : `${Math.round(x * 100)}%`;
}

function timeAgo(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  // Clamp at 0 so minor client/server clock skew never reads as negative.
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v != null) node.setAttribute(k, v);
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function modeBadge(mode) {
  return el("span", { class: `badge ${mode || "none"}` }, mode || "no prolific");
}

// ── sidebar (study list) ───────────────────────────────────────────────
function renderSidebar() {
  const sidebar = $("sidebar");
  sidebar.replaceChildren();
  if (!state.sessions.length) {
    sidebar.append(el("div", { class: "placeholder" }, "No live studies found in the data tree yet."));
    return;
  }
  for (const s of state.sessions) {
    const target = s.target_participants || 0;
    const filled = target ? Math.min(1, s.n_with_data / target) : 0;
    const card = el(
      "div",
      {
        class: `card${s.collection_session_id === state.selected ? " selected" : ""}${s.has_warning ? " warn" : ""}`,
        onclick: () => select(s.collection_session_id),
      },
      el("div", { class: "card-title" }, s.experiment_id),
      el(
        "div",
        { class: "card-row" },
        modeBadge(s.prolific_mode),
        s.has_warning ? el("span", { class: "chip-warn" }, "⚠ data check") : null,
      ),
      el(
        "div",
        { class: `progress${s.has_warning ? " warn" : ""}` },
        el("span", { style: `width:${Math.round(filled * 100)}%` }),
      ),
      el(
        "div",
        { class: "card-meta" },
        `${s.n_with_data}${target ? " / " + target : ""} done · ${s.n_responses} submitted · last ${timeAgo(s.last_submission_at)}`,
      ),
    );
    sidebar.append(card);
  }
}

// ── detail view ────────────────────────────────────────────────────────
function statCard(label, value, opts = {}) {
  const valEl = el("div", { class: `value${opts.warn ? " warn" : ""}` });
  valEl.innerHTML = value;
  return el("div", { class: "stat" }, el("div", { class: "label" }, label), valEl);
}

function renderProlific(p) {
  if (!p || (!p.study_id && !p.error)) {
    return el("div", { class: "muted" }, "No Prolific study attached to this session.");
  }
  if (p.error) {
    return el("div", { class: "warn-banner" }, `Prolific status unavailable: ${p.error}`);
  }
  const counts = p.counts || {};
  const order = ["ACTIVE", "AWAITING_REVIEW", "APPROVED", "RETURNED", "TIMED_OUT", "REJECTED"];
  const keys = order.filter((k) => k in counts).concat(Object.keys(counts).filter((k) => !order.includes(k)));
  const items = keys.map((k) =>
    el("span", { class: "badge", style: "margin-right:6px" }, `${k.toLowerCase().replace(/_/g, " ")}: ${counts[k]}`),
  );
  return el(
    "div",
    {},
    el("div", { style: "margin-bottom:8px" }, `Study status: `, el("strong", {}, p.status || "unknown"),
      p.places_taken != null ? ` · ${p.places_taken}/${p.places_total ?? "?"} places taken` : ""),
    el("div", {}, ...items),
  );
}

function renderBalance(b) {
  if (!b || !b.total_valid_trials) {
    return el("div", { class: "muted" }, "No completed trials yet.");
  }
  const leftPct = Math.round((b.p_left || 0) * 100);
  const split = el(
    "div",
    { class: "split" },
    el("div", { class: "l", style: `width:${leftPct}%` }, leftPct >= 12 ? `left ${leftPct}%` : ""),
    el("div", { class: "r", style: `width:${100 - leftPct}%` }, 100 - leftPct >= 12 ? `right ${100 - leftPct}%` : ""),
  );
  return el(
    "div",
    {},
    b.warning ? el("div", { class: "warn-banner" }, `⚠ ${b.warning}`) : null,
    split,
    el(
      "div",
      { class: "split-legend" },
      el("span", {}, `${b.n_left} chose left`),
      el("span", {}, `${b.total_valid_trials} valid trials`),
      el("span", {}, `${b.n_right} chose right`),
    ),
  );
}

function renderParticipants(participants) {
  if (!participants.length) {
    return el("div", { class: "muted" }, "No participants have submitted yet.");
  }
  const head = el(
    "tr",
    {},
    el("th", {}, "Participant"),
    el("th", {}, "Trials"),
    el("th", {}, "Left vs right"),
    el("th", {}, "P(left)"),
    el("th", {}, "Submitted"),
    el("th", {}, ""),
  );
  const rows = participants.map((p) => {
    const leftPct = p.p_left == null ? 0 : Math.round(p.p_left * 100);
    const minibar = el("span", { class: "minibar", title: `${p.n_left} left / ${p.n_right} right` },
      el("span", { style: `width:${leftPct}%` }));
    const trials = p.n_valid_trials === p.n_trials
      ? String(p.n_trials)
      : `${p.n_valid_trials} / ${p.n_trials}`;
    return el(
      "tr",
      { class: p.degenerate ? "bad" : "" },
      el("td", { class: "pid" }, p.prolific_pid || p.participant_id),
      el("td", {}, trials),
      el("td", {}, minibar),
      el("td", {}, pct(p.p_left)),
      el("td", { class: "muted" }, timeAgo(p.submitted_at)),
      el("td", {}, p.degenerate ? el("span", { class: "flag", title: "Chose one side on every trial" }, "⚠ one-sided") : ""),
    );
  });
  return el("table", { class: "participants" }, el("thead", {}, head), el("tbody", {}, ...rows));
}

function renderDetail() {
  const content = $("content");
  const d = state.detail;
  if (!d) {
    content.replaceChildren(el("div", { class: "placeholder" }, "Select a study to see live participant data."));
    return;
  }
  const target = d.target_participants || 0;

  const urlNode = d.experiment_url
    ? el("a", { href: d.experiment_url, target: "_blank", rel: "noopener" }, d.experiment_url)
    : "no live URL";

  content.replaceChildren(
    el(
      "div",
      { class: "detail-head" },
      el("h1", {}, d.experiment_id),
      modeBadge(d.prolific_mode),
      d.has_warning ? el("span", { class: "chip-warn" }, "⚠ check data quality") : null,
    ),
    el("div", { class: "detail-sub" }, d.collection_session_id, " · ", urlNode),

    el(
      "div",
      { class: "stat-grid" },
      statCard("Completed", `${d.n_with_data}${target ? ` <small>/ ${target}</small>` : ""}`),
      statCard("Submitted", String(d.n_responses)),
      statCard("Overall P(left)", pct(d.overall_p_left), { warn: d.choice_balance && d.choice_balance.is_degenerate }),
      statCard("One-sided participants", String(d.n_degenerate_participants), { warn: d.n_degenerate_participants > 0 }),
    ),

    el(
      "div",
      { class: `panel${d.choice_balance && d.choice_balance.is_degenerate ? " warn" : ""}` },
      el("h2", {}, "Data quality · choice balance"),
      renderBalance(d.choice_balance),
    ),

    el("div", { class: "panel" }, el("h2", {}, "Prolific recruitment"), renderProlific(d.prolific)),

    el("div", { class: "panel" }, el("h2", {}, `Participants (${d.participants.length})`), renderParticipants(d.participants)),
  );
}

// ── data flow ──────────────────────────────────────────────────────────
async function refresh() {
  try {
    const body = await getJSON("/api/sessions");
    state.sessions = body.sessions;
    renderSidebar();
    if (state.selected) {
      state.detail = await getJSON(`/api/session/${encodeURIComponent(state.selected)}`);
      renderDetail();
    }
    setRefreshState(`updated ${new Date().toLocaleTimeString()}`);
  } catch (err) {
    setRefreshState(`error: ${err.message}`);
  }
}

async function select(id) {
  state.selected = id;
  renderSidebar();
  try {
    state.detail = await getJSON(`/api/session/${encodeURIComponent(id)}`);
  } catch (err) {
    state.detail = null;
    $("content").replaceChildren(el("div", { class: "warn-banner" }, `Failed to load session: ${err.message}`));
    return;
  }
  renderDetail();
}

function setRefreshState(text) {
  $("refresh-state").textContent = text;
}

// ── auto-refresh loop ──────────────────────────────────────────────────
function tick() {
  if (!$("autorefresh").checked) return;
  state.countdown -= 1;
  if (state.countdown <= 0) {
    state.countdown = REFRESH_MS / 1000;
    refresh();
  }
}

function init() {
  $("refresh-now").addEventListener("click", () => {
    state.countdown = REFRESH_MS / 1000;
    refresh();
  });
  $("autorefresh").addEventListener("change", (e) => {
    if (e.target.checked) {
      state.countdown = REFRESH_MS / 1000;
      refresh();
    }
  });
  refresh();
  setInterval(tick, 1000);
}

init();

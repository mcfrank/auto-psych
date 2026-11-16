"""De-identified trial data from the live reference-game studies, for sharing (PI 2026-11-15).

Every live study of the closed and open campaigns stored each submission with its Prolific ids
(`private/raw_collected/experiment<n>.json`, on Sherlock only). This module turns those into tables
that can be committed:

- **people** get a keyed hash of their Prolific id (`person_hash`: HMAC-SHA256 under a secret key
  that lives only with the Prolific ids, on Sherlock and the PI's machine). The same person has the
  same hash in every study, so repeat participation stays visible; without the key, nobody holding
  a Prolific id can find that person in the data;
- **free text is dropped** (the open campaign's end-of-study `comment`);
- every other field is kept only if it is on a whitelist (`TRIAL_KEEP`): a field the whitelist does
  not name raises, so a new page's data are looked at before anything of them is shared;
- `check_no_ids` refuses a table in which any of the studies' real identifiers appears.

The run-local pseudonym already used in the committed records (`live<k>` in the closed campaign,
`p<k>` in the open one) is kept beside the hash, so the shared trials join the committed aggregates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

# Prolific identifiers on each stored submission; none is kept.
ID_FIELDS = ("participant_id_str", "prolific_pid", "prolific_session_id", "prolific_study_id")
# Per-submission fields kept on the people table.
PERSON_KEEP = ("submitted_at_client", "list_index")
# Dropped from every trial record: the Prolific id (`participant_id`), free text (`comment`), and
# the page's rendered HTML (`stimulus`: the display is in the structured fields, the page in the
# committed designs) and the plugin's version, which say nothing about the person.
TRIAL_DROP = frozenset({"participant_id", "comment", "stimulus", "plugin_version"})
TRIAL_KEEP = frozenset({
    # jsPsych bookkeeping and timing
    "trial_type", "trial_index", "time_elapsed", "rt", "task", "phase", "trial_number", "n_test_trials",
    # list and design provenance
    "list_index", "list_seed", "list_assignment", "design_name", "design_sha256", "paradigm_sha256",
    # the display
    "condition", "is_catch", "catch_target", "spec_index", "item", "feature_names", "objects", "roles",
    "utterance", "word", "words", "query", "messages", "display_order", "bases", "unit", "heard", "probe",
    "second_probe", "target", "expect", "domain",
    # the response (button choices, ratings, production choices: no typed text)
    "response", "choice_position", "choice", "chosen_alt", "check", "correct", "attempt", "rating",
})
PROLIFIC_LIKE = re.compile(r"[0-9a-f]{24}")


class UnknownField(ValueError):
    """A trial record has a field the whitelist does not name: look at it before sharing."""


def person_hash(key: bytes, prolific_id: str, n_hex: int = 16) -> str:
    """The shared id of a person: HMAC-SHA256 of their Prolific id under the secret key."""
    if not key:
        raise ValueError("no key: the hash must be keyed, or anyone with a Prolific id could find its person")
    return "h" + hmac.new(key, prolific_id.encode(), hashlib.sha256).hexdigest()[:n_hex]


def _cell(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else json.dumps(value, sort_keys=True)


def deidentify_trials(records: Sequence[Mapping[str, Any]], person: str) -> List[Dict[str, Any]]:
    """One person's jsPsych records, whitelisted, with their shared id."""
    out = []
    for rec in records:
        unknown = set(rec) - TRIAL_KEEP - TRIAL_DROP
        if unknown:
            raise UnknownField(f"trial fields not on the whitelist: {sorted(unknown)} (task {rec.get('task')!r})")
        out.append({"person": person, **{k: _cell(v) for k, v in rec.items() if k in TRIAL_KEEP}})
    return out


def submission_ids(response: Mapping[str, Any], records: Iterable[Mapping[str, Any]] = ()) -> Set[str]:
    """Every identifier a stored submission carries (to check none survives)."""
    ids = {str(response[f]) for f in ID_FIELDS if response.get(f)}
    ids |= {str(r["participant_id"]) for r in records if r.get("participant_id")}
    return ids


def check_no_ids(rows: Iterable[Mapping[str, Any]], ids: Set[str]) -> None:
    """Raise if any real identifier, or anything shaped like a Prolific id, is in a table."""
    ids = {i for i in ids if i}
    for n, row in enumerate(rows):
        text = json.dumps(row, sort_keys=True, default=str)
        hits = set(PROLIFIC_LIKE.findall(text))
        if ids & hits or any(i in text for i in ids if not PROLIFIC_LIKE.fullmatch(i)):
            raise ValueError(f"row {n} of a shared table contains a Prolific identifier")
        if any(re.search(rf"(?<![0-9a-f]){re.escape(h)}(?![0-9a-f])", text) for h in hits):
            raise ValueError(f"row {n} of a shared table contains a Prolific-shaped id (24 hex characters)")


def pseudonym_of(id_map: Mapping[str, Any], prolific_id: str) -> Optional[str]:
    """The run's own pseudonym: a closed run's map holds ``live<k>``, the open campaign's ``{"id": p<k>}``."""
    entry = id_map.get(prolific_id)
    return entry.get("id") if isinstance(entry, Mapping) else entry

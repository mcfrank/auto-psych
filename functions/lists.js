/**
 * Pure helpers for list assignment and JSON results (no Firebase imports, so
 * they are testable with plain node: tests/test_functions_lists.py).
 *
 * List assignment (the RSA live experiments, 2026-10-10): every participant of
 * a collection session gets the next trial list in arrival order, round robin
 * over the session's n_lists. A participant who reloads gets the list they
 * were given. Each list is a balanced subset of the design, so assigning lists
 * in order keeps every designed display's response count equal; a participant
 * who abandons leaves a gap, and lists past n_lists start again at 0.
 */

function validateAssign(body) {
  if (!body || typeof body !== "object") return "Missing body";
  if (!body.collection_session_id) return "Missing collection_session_id";
  const n = Number(body.n_lists);
  if (!Number.isInteger(n) || n < 1) return "n_lists must be a positive integer";
  if (!body.participant_key || typeof body.participant_key !== "string") return "Missing participant_key";
  return "";
}

/**
 * The list for a participant, given the session's stored state
 * ({n_lists, next} or null) and the participant's earlier assignment
 * ({list_index} or null). Returns {listIndex, nextState} where nextState is
 * null when nothing changes (the participant already had a list).
 */
function assignmentPlan(stored, prior, nLists) {
  if (stored && stored.n_lists !== nLists) {
    throw new Error(`this session assigns over ${stored.n_lists} lists, not ${nLists}`);
  }
  if (prior && Number.isInteger(prior.list_index)) {
    return { listIndex: prior.list_index, nextState: null };
  }
  const next = stored ? stored.next : 0;
  return { listIndex: next % nLists, nextState: { n_lists: nLists, next: next + 1 } };
}

/** Every stored response of a session, whole, for a domain whose trials are not the default CSV's. */
function responsesToJson(docs) {
  return docs.map((doc) => {
    const d = doc.data();
    return {
      participant_id_str: doc.id,
      prolific_pid: d.prolific_pid || null,
      prolific_study_id: d.prolific_study_id || null,
      prolific_session_id: d.prolific_session_id || null,
      submitted_at_client: d.submitted_at_client || null,
      list_index: d.list_index == null ? null : d.list_index,
      trials: Array.isArray(d.trials) ? d.trials : [],
    };
  });
}

module.exports = { validateAssign, assignmentPlan, responsesToJson };

/**
 * Firebase Cloud Functions for auto-psych deployment V1.
 *
 * POST /register_session (admin token) marks a collection session as active.
 * POST /submit stores participant trial data under a REGISTERED session.
 * GET /results (admin token) exports collection-session responses as CSV.
 *
 * Auth model: the deployment pipeline holds a shared secret (RESULTS_TOKEN,
 * provisioned via functions/.env at deploy time). /results and
 * /register_session require it in the `x-results-token` header — without it,
 * anyone who loads the public experiment page (which necessarily carries the
 * collection_session_id) could read every participant's data. /submit stays
 * public (participants' browsers call it) but only accepts sessions the
 * pipeline registered, so drive-by posts cannot fabricate data under made-up
 * sessions; the legacy project_id/run_id write path is gone for the same
 * reason.
 */
const crypto = require("crypto");
const functions = require("firebase-functions/v1");
const admin = require("firebase-admin");

admin.initializeApp();

const db = admin.firestore();
const region = process.env.FUNCTION_REGION || "us-central1";
const https = functions.region(region).https;

function parseBody(req) {
  if (typeof req.body === "string") {
    return JSON.parse(req.body);
  }
  return req.body || {};
}

function legacyRunKey(projectId, runId) {
  return `${projectId}_${runId}`;
}

/**
 * Enforce the admin token. Fails LOUDLY (500) when the function itself has no
 * RESULTS_TOKEN configured — an unconfigured deployment must never fall open.
 * Returns true when the request may proceed.
 */
function requireAdminToken(req, res) {
  const expected = process.env.RESULTS_TOKEN || "";
  if (!expected) {
    console.error("RESULTS_TOKEN is not configured on this function");
    res.status(500).send("Server misconfigured: RESULTS_TOKEN not set");
    return false;
  }
  const got = String(req.get("x-results-token") || "");
  const a = Buffer.from(expected);
  const b = Buffer.from(got);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    res.status(403).send("Forbidden");
    return false;
  }
  return true;
}

function validatePayload(body) {
  if (!Array.isArray(body.trials)) {
    return "Missing trials array";
  }
  if (!body.collection_session_id) {
    return "Missing collection_session_id (legacy project_id/run_id submissions are no longer accepted)";
  }
  return "";
}

function csvValue(value) {
  const s = value == null ? "" : String(value);
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function responsesToCsv(docs) {
  const rows = [];
  let participantIndex = 0;
  docs.forEach((doc) => {
    const participantIdStr = doc.id;
    const data = doc.data();
    const trials = Array.isArray(data.trials) ? data.trials : [];
    trials.forEach((trial, trialIndex) => {
      if (
        trial.sequence_a == null ||
        trial.sequence_b == null ||
        trial.chose_left == null
      ) {
        return;
      }
      const choseLeft = trial.chose_left ? 1 : 0;
      rows.push({
        participant_id: participantIndex,
        participant_id_str: participantIdStr,
        trial_index: trialIndex,
        sequence_a: String(trial.sequence_a),
        sequence_b: String(trial.sequence_b),
        chose_left: choseLeft,
        chose_right: choseLeft ? 0 : 1,
        model: "",
      });
    });
    participantIndex += 1;
  });
  const fields = [
    "participant_id",
    "participant_id_str",
    "trial_index",
    "sequence_a",
    "sequence_b",
    "chose_left",
    "chose_right",
    "model",
  ];
  const lines = [
    fields.join(","),
    ...rows.map((row) => fields.map((field) => csvValue(row[field])).join(",")),
  ];
  return lines.join("\n");
}

exports.register_session = https.onRequest(async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).send("Method Not Allowed");
    return;
  }
  if (!requireAdminToken(req, res)) {
    return;
  }

  let body;
  try {
    body = parseBody(req);
  } catch {
    res.status(400).send("Invalid JSON");
    return;
  }
  const sessionId = String(body.collection_session_id || "");
  if (!sessionId) {
    res.status(400).send("Missing collection_session_id");
    return;
  }

  try {
    await db.collection("collection_sessions").doc(sessionId).set(
      {
        collection_session_id: sessionId,
        project_id: body.project_id || null,
        run_id: body.run_id || null,
        deployment_id: body.deployment_id || null,
        revoked: false,
        registered_at: admin.firestore.FieldValue.serverTimestamp(),
      },
      { merge: true },
    );
    res.status(200).send("OK");
  } catch (err) {
    console.error(err);
    res.status(500).send("Write failed");
  }
});

exports.submit = https.onRequest(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Headers", "Content-Type");
    res.status(204).send("");
    return;
  }
  if (req.method !== "POST") {
    res.status(405).send("Method Not Allowed");
    return;
  }

  let body;
  try {
    body = parseBody(req);
  } catch {
    res.status(400).send("Invalid JSON");
    return;
  }

  const validationError = validatePayload(body);
  if (validationError) {
    res.status(400).send(validationError);
    return;
  }

  const sessionId = String(body.collection_session_id);
  const sessionRef = db.collection("collection_sessions").doc(sessionId);

  const participantId = String(body.participant_id || body.prolific_pid || Date.now());
  const created = admin.firestore.FieldValue.serverTimestamp();
  const record = {
    project_id: body.project_id || null,
    experiment_id: body.experiment_id || null,
    run_id: body.run_id || null,
    study_id: body.study_id || null,
    deployment_id: body.deployment_id || null,
    collection_session_id: sessionId,
    agent_backend: body.agent_backend || null,
    collection_owner: body.collection_owner || null,
    firebase_project: body.firebase_project || null,
    prolific_mode: body.prolific_mode || null,
    prolific_study_id: body.prolific_study_id || body.prolific_study_id_from_url || null,
    prolific_pid: body.prolific_pid || null,
    prolific_session_id: body.prolific_session_id || null,
    participant_id: participantId,
    consented_at: body.consented_at || null,
    submitted_at_client: body.submitted_at_client || null,
    user_agent: body.user_agent || null,
    trials: body.trials,
    created_at: created,
  };

  try {
    // Only sessions the deployment pipeline registered accept data — a
    // drive-by POST with a fabricated session id must not create rows.
    const session = await sessionRef.get();
    if (!session.exists || session.get("revoked") === true) {
      res.status(403).send("Unknown or revoked collection session");
      return;
    }

    await sessionRef.collection("responses").doc(participantId).set(record, { merge: true });
    await db.collection("participants").doc(participantId).set(
      {
        participant_id: participantId,
        latest_collection_session_id: sessionId,
        latest_deployment_id: body.deployment_id || null,
        latest_project_id: body.project_id || null,
        updated_at: created,
      },
      { merge: true },
    );
    res.status(200).send("OK");
  } catch (err) {
    console.error(err);
    res.status(500).send("Write failed");
  }
});

exports.results = https.onRequest(async (req, res) => {
  if (req.method !== "GET") {
    res.status(405).send("Method Not Allowed");
    return;
  }
  if (!requireAdminToken(req, res)) {
    return;
  }

  const collectionSessionId = req.query.collection_session_id;
  const runId = req.query.run_id;
  const projectId = req.query.project_id;

  let ref;
  if (collectionSessionId) {
    ref = db
      .collection("collection_sessions")
      .doc(String(collectionSessionId))
      .collection("responses");
  } else if (runId && projectId) {
    // Legacy read path for pre-session-id deployments (token-guarded).
    ref = db.collection("runs").doc(legacyRunKey(projectId, runId)).collection("responses");
  } else {
    res.status(400).send("Missing collection_session_id or project_id/run_id");
    return;
  }

  try {
    const snap = await ref.get();
    res.set("Content-Type", "text/csv");
    res.send(responsesToCsv(snap.docs));
  } catch (err) {
    console.error(err);
    res.status(500).send("Read failed");
  }
});

/**
 * Firebase Cloud Functions for auto-psych deployment V1.
 *
 * POST /submit stores participant trial data under a collection session.
 * GET /results exports collection-session responses as CSV for the pipeline.
 */
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

function validatePayload(body) {
  if (!Array.isArray(body.trials)) {
    return "Missing trials array";
  }
  if (body.collection_session_id) {
    return "";
  }
  if (body.project_id && body.run_id) {
    return "";
  }
  return "Missing collection_session_id or project_id/run_id";
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

  const participantId = String(body.participant_id || body.prolific_pid || Date.now());
  const created = admin.firestore.FieldValue.serverTimestamp();
  const record = {
    project_id: body.project_id || null,
    experiment_id: body.experiment_id || null,
    run_id: body.run_id || null,
    study_id: body.study_id || null,
    deployment_id: body.deployment_id || null,
    collection_session_id: body.collection_session_id || null,
    agent_backend: body.agent_backend || null,
    collection_owner: body.collection_owner || null,
    firebase_project: body.firebase_project || null,
    prolific_mode: body.prolific_mode || null,
    prolific_study_id: body.prolific_study_id || body.prolific_study_id_from_url || null,
    prolific_pid: body.prolific_pid || null,
    prolific_session_id: body.prolific_session_id || null,
    participant_id: participantId,
    submitted_at_client: body.submitted_at_client || null,
    user_agent: body.user_agent || null,
    trials: body.trials,
    created_at: created,
  };

  try {
    if (body.collection_session_id) {
      await db
        .collection("collection_sessions")
        .doc(String(body.collection_session_id))
        .collection("responses")
        .doc(participantId)
        .set(record, { merge: true });
      await db.collection("participants").doc(participantId).set(
        {
          participant_id: participantId,
          latest_collection_session_id: body.collection_session_id,
          latest_deployment_id: body.deployment_id || null,
          latest_project_id: body.project_id || null,
          updated_at: created,
        },
        { merge: true },
      );
    } else {
      const runKey = legacyRunKey(body.project_id, body.run_id);
      await db
        .collection("runs")
        .doc(runKey)
        .collection("responses")
        .doc(participantId)
        .set(record, { merge: true });
    }
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

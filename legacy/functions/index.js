/**
 * Firebase Cloud Functions: POST /submit (store trial data), GET /results (export CSV).
 * Hosting rewrites /submit and /results to these functions (see firebase.json).
 */
const functions = require("firebase-functions/v1");
const admin = require("firebase-admin");

admin.initializeApp();

const db = admin.firestore();

/**
 * POST /submit — body: { run_id, project_id, participant_id?, trials }.
 * Writes to Firestore: runs/{project_id}_{run_id}/responses/{participant_id}
 */
exports.submit = functions.https.onRequest((req, res) => {
  if (req.method !== "POST") {
    res.status(405).send("Method Not Allowed");
    return;
  }
  let body;
  try {
    body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
  } catch {
    res.status(400).send("Invalid JSON");
    return;
  }
  const { run_id, project_id, participant_id, trials } = body;
  if (!run_id || !project_id || !Array.isArray(trials)) {
    res.status(400).send("Missing run_id, project_id, or trials");
    return;
  }
  const docId = String(participant_id ?? Date.now());
  const runKey = `${project_id}_${run_id}`;
  const ref = db.collection("runs").doc(runKey).collection("responses").doc(docId);
  ref
    .set({ trials, created: admin.firestore.FieldValue.serverTimestamp() })
    .then(() => {
      res.status(200).send("OK");
    })
    .catch((err) => {
      console.error(err);
      res.status(500).send("Write failed");
    });
});

/**
 * GET /results?run_id=...&project_id=...
 * Reads all docs under runs/{project_id}_{run_id}/responses, flattens to CSV.
 */
exports.results = functions.https.onRequest((req, res) => {
  if (req.method !== "GET") {
    res.status(405).send("Method Not Allowed");
    return;
  }
  const run_id = req.query.run_id;
  const project_id = req.query.project_id;
  if (!run_id || !project_id) {
    res.status(400).send("Missing run_id or project_id");
    return;
  }
  const runKey = `${project_id}_${run_id}`;
  const ref = db.collection("runs").doc(runKey).collection("responses");
  ref
    .get()
    .then((snap) => {
      const rows = [];
      let participant_id = 0;
      snap.docs.forEach((d) => {
        const participant_id_str = d.id;
        const data = d.data();
        const trials = data.trials || [];
        trials.forEach((trial, i) => {
          if (
            trial.sequence_a != null &&
            trial.sequence_b != null &&
            trial.chose_left != null
          ) {
            rows.push({
              participant_id,
              participant_id_str,
              trial_index: i,
              sequence_a: String(trial.sequence_a),
              sequence_b: String(trial.sequence_b),
              chose_left: trial.chose_left ? 1 : 0,
              chose_right: trial.chose_left ? 0 : 1,
              model: "",
            });
          }
        });
        participant_id += 1;
      });
      // Output CSV (participant_id_str = Firestore doc id, so client can filter by run batch)
      const header = "participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model";
      const lines = [header, ...rows.map((r) =>
        [r.participant_id, r.participant_id_str, r.trial_index, r.sequence_a, r.sequence_b, r.chose_left, r.chose_right, r.model].join(",")
      )];
      res.set("Content-Type", "text/csv");
      res.send(lines.join("\n"));
    })
    .catch((err) => {
      console.error(err);
      res.status(500).send("Read failed");
    });
});

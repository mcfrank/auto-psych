# Design Rationale

- **Number of stimuli**: 32 pairs were selected for the experiment, fulfilling the length constraint (~4 minutes of judgments).
- **Candidate pool**: A diverse set of 300 candidates was sampled uniformly from the space of all possible coin-flip sequences of length 2 to 8. This ensures tractable EIG evaluation while still sampling heavily across possible lengths and structures.
- **EIG Range**: The selected 32 stimuli have Expected Information Gain (EIG) scores ranging from ~0.21 to ~0.40. 
- **Model discrimination**: By selecting stimuli with high EIG, the design inherently prefers sequence pairs for which the active cognitive models in `model_registry.yaml` make the most divergent predictions. For example, comparing very short strings (e.g. "TH") against longer, highly structured or streaky strings (e.g. "TTTTHHTH") likely forces models relying on alternation vs. length vs. run-length penalties to disagree, maximizing the information gained from participant responses.

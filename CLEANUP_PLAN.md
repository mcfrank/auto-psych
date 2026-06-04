# Cleanup Plan

Current state:
- Active orchestration lives under `src/pipelines/`.
- Shared support code has been split into `src/runtime/`, `src/experiments/`, `src/models/`, `src/registry/`, and `src/validation/`.
- Retired LangGraph/LangChain code, old prompts, Cloud Run, SLURM, and legacy tests are in `legacy/`.

## Immediate Cleanup

1. Remove or quarantine the remaining compatibility wrappers in `src/models/`.
2. Replace any remaining old import paths with the new package paths.
3. Decide whether `src/models/__init__.py` should stay as a thin public API or be reduced further.

## Next Targets

1. Split `src/eig/` into a smaller package if it still mixes model inspection, annotation, and reporting concerns.
2. Split `src/stats/` if it grows beyond a single correlation module.
3. Review whether `src/validation/stages/` should stay as stage-local files or be folded into pipeline-local validators.

## Compatibility Removal

1. Delete the shim modules in `src/models/` once all imports are migrated.
2. Delete any remaining `src/agents`-style legacy wrappers if new references appear.
3. Remove dead compatibility paths from tests and docs once the tree is stable.

## Final Hardening

1. Run `compileall` and the active pytest set after each move.
2. Keep `legacy/` read-only unless old behavior must be referenced.
3. Prefer moving code into explicit ownership folders rather than adding generic utility modules.

## Success Criteria

- No active code imports retired module paths.
- The active tree is readable by ownership domain, not by historical accident.
- Compatibility wrappers are temporary and intentionally small.
- Anything not needed by the active pipelines is either in `legacy/` or deleted.

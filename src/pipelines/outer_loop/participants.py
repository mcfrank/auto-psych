"""Participant-model backends for simulated-participant data collection.

A *participant model* answers one experiment trial: given a system prompt (the
participant persona/instructions) and a user message (the rendered stimulus), it
returns the model's raw text reply. The collect step parses that reply into a
choice (see ``collect._parse_participant_answer``).

Two backends share the one ``ParticipantModel`` interface so the rest of the
pipeline never needs to know which is in use (and so the future browser
``simulated_participants`` path can reuse the same abstraction):

- ``"closed"``: hosted API models. Default is the project's Gemini client
  (``outer_loop.llm``). Light dependencies — always available.
- ``"open"``: local Hugging Face ``transformers`` models, selected by hub id.
  Heavy dependencies (``torch``, ``transformers``, ``accelerate``) are an
  *optional* install (``uv sync --group open-models``) and are imported lazily,
  so the base pipeline never needs them.

Resolve a backend with :func:`get_participant_model`.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
PARTICIPANT_BACKENDS = (CLOSED, OPEN)

# Default Hugging Face model for the open backend. ~7.6B params (~15 GB in bf16)
# — needs a GPU or a lot of RAM. Override with a smaller id (e.g.
# Qwen/Qwen2.5-0.5B-Instruct) for quick local checks.
DEFAULT_OPEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@runtime_checkable
class ParticipantModel(Protocol):
    """Anything that can answer a single trial as a participant."""

    name: str

    def answer(self, system: str, user: str) -> str:
        """Return the model's raw text reply to one trial."""
        ...


# ─────────────────────────────────────────────
# Closed (hosted API) backend
# ─────────────────────────────────────────────


class ClosedParticipantModel:
    """Hosted-API participant. Wraps the outer-loop Gemini client.

    ``model`` overrides the default closed model id; ``None`` uses the client
    default. The client is constructed once and reused across trials.

    Reproducibility caveat: the hosted Gemini API does not expose a sampling seed
    that guarantees deterministic output, so data collected through this backend
    is NOT bit-reproducible run-to-run (unlike the open backend, which is seeded,
    and the synthetic PyMC paths). Record the model id and treat each collection
    as a fresh sample.
    """

    # Bound each participant API call so a single stuck request can't hang the
    # whole collection (a normal reply takes a few seconds).
    REQUEST_TIMEOUT_SEC = 60

    def __init__(self, model: Optional[str] = None) -> None:
        from src.pipelines.outer_loop.llm import get_llm

        self.name = f"closed:{model}" if model else "closed:default"
        # Raises if credentials / model are unavailable — surfaced to the caller.
        self._llm = get_llm(model=model, timeout=self.REQUEST_TIMEOUT_SEC)

    def answer(self, system: str, user: str) -> str:
        from src.pipelines.outer_loop.llm import invoke_llm

        return invoke_llm(
            system=system, user=user, llm=self._llm, timeout=self.REQUEST_TIMEOUT_SEC
        )


# ─────────────────────────────────────────────
# Open (Hugging Face transformers) backend
# ─────────────────────────────────────────────


def _import_hf():
    """Import torch + transformers lazily, with an actionable error if absent."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the group
        raise ImportError(
            "Open participant models need `torch` and `transformers`, which are "
            "not part of the base install. Add the optional group:\n"
            "    uv sync --group open-models        # CPU / Apple-MPS torch\n"
            "For an NVIDIA GPU, first install the torch wheel matching your CUDA "
            "version from https://pytorch.org/get-started/locally/, then run the "
            "sync above."
        ) from exc
    return torch, AutoTokenizer, AutoModelForCausalLM


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class OpenParticipantModel:
    """Local Hugging Face causal-LM participant, selected by hub id.

    The model + tokenizer load once. ``answer`` applies the tokenizer's chat
    template when present (falling back to a plain system+user concatenation),
    samples a short continuation, and returns the newly generated text only.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        max_new_tokens: int = 24,
        temperature: float = 0.7,
        seed: int = 0,
    ) -> None:
        if not model_name:
            raise ValueError(
                "open participant backend requires a Hugging Face model id "
                "(pass --hf-model <hub id>)"
            )
        torch, AutoTokenizer, AutoModelForCausalLM = _import_hf()
        self.name = f"open:{model_name}"
        self._torch = torch
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device = device or _auto_device(torch)
        # Seed torch's global RNG once so the sequence of sampled generations is
        # reproducible run-to-run. ``answer`` is called once per trial in a fixed
        # order, each draw advancing the same RNG, so a fixed seed makes the whole
        # collected dataset reproducible (do_sample=True is otherwise unseeded —
        # the data that drives modeling would change on every run).
        torch.manual_seed(seed)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        ).to(self._device)
        self._model.eval()

    def _build_prompt(self, system: str, user: str) -> str:
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Some chat templates reject a system role; fall back to a plain
                # concatenation. Log it (rather than swallowing silently) — an
                # un-templated prompt makes an instruction-tuned model follow the
                # answer format far less reliably, inflating unparseable replies,
                # so the degradation must be visible in the run log.
                logger.warning(
                    "chat template for %s rejected the system+user messages; "
                    "falling back to plain prompt concatenation",
                    self.name,
                    exc_info=True,
                )
        return f"{system}\n\n{user}\n"

    def answer(self, system: str, user: str) -> str:
        torch = self._torch
        prompt = self._build_prompt(system, user)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────


def get_participant_model(
    backend: str, model_name: Optional[str] = None
) -> ParticipantModel:
    """Resolve a participant-model backend.

    ``backend``: ``"closed"`` (hosted API; ``model_name`` optionally overrides the
    closed model id) or ``"open"`` (Hugging Face; ``model_name`` is the required
    hub id). Raises ``ValueError`` on an unknown backend.
    """
    if backend == CLOSED:
        return ClosedParticipantModel(model_name)
    if backend == OPEN:
        return OpenParticipantModel(model_name)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown participant backend {backend!r}; expected one of {PARTICIPANT_BACKENDS}"
    )

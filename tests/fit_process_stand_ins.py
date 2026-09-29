"""Stand-ins for the child process of ``pymc_inference.sample_fits_time_limited``.

They live in their own module because the child is a spawned interpreter: its
entry point must be importable by name (``tests.fit_process_stand_ins``).
Each has the signature of ``_sample_in_own_session``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def sample_with_a_chain_forever(name, models_dir, responses_path, settings, cache_dir, sender):
    """Start a session, fork a 'chain', leave a half-written fit, and never
    finish; the pids go to ``cache_dir/pids`` for the test to check."""
    os.setsid()
    chain = os.fork()
    if chain == 0:
        time.sleep(600)
        os._exit(0)
    (Path(cache_dir) / f".{name}.0123.nc.{os.getpid()}.partial").write_text("half")
    (Path(cache_dir) / "pids").write_text(f"{os.getpid()} {chain}")
    time.sleep(600)


def die_without_reporting(name, models_dir, responses_path, settings, cache_dir, sender):
    os._exit(3)


def fail_as_the_model(name, models_dir, responses_path, settings, cache_dir, sender):
    sender.send(("model", "ValueError: the model's own failure"))


def write_the_fit(name, models_dir, responses_path, settings, cache_dir, sender):
    """Persist a stand-in fit where the parent expects it, and report success."""
    from src.models.pymc_inference import cached_fit_path, fit_fingerprint

    fingerprint = fit_fingerprint(name, Path(models_dir), Path(responses_path), settings)
    cached_fit_path(Path(cache_dir), name, fingerprint).write_text("fit")
    sender.send(("ok", ""))


def write_the_fit_and_hang(name, models_dir, responses_path, settings, cache_dir, sender):
    """Persist a stand-in fit, then never report."""
    from src.models.pymc_inference import cached_fit_path, fit_fingerprint

    fingerprint = fit_fingerprint(name, Path(models_dir), Path(responses_path), settings)
    cached_fit_path(Path(cache_dir), name, fingerprint).write_text("fit")
    time.sleep(600)


def import_arviz_and_write_the_fit(name, models_dir, responses_path, settings, cache_dir, sender):
    """Import arviz (which writes its once-a-day marker under the cache
    directory), then persist a stand-in fit and report success."""
    import arviz  # noqa: F401

    write_the_fit(name, models_dir, responses_path, settings, cache_dir, sender)


def import_arviz_and_name_the_cache_dir() -> str:
    """For a fit-pool worker: import arviz, then return the cache directory it
    wrote its marker under."""
    import arviz  # noqa: F401

    return os.environ["XDG_CACHE_HOME"]

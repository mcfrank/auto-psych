"""The RSA fit cache is keyed by the fingerprint, not by the model's name."""

from pathlib import Path

from src.rsa.loop.fitting import cache_paths


def test_fits_with_different_fingerprints_get_different_files(tmp_path):
    a = cache_paths(tmp_path, "rsa_l1", "0123456789abcdef0123")
    b = cache_paths(tmp_path, "rsa_l1", "fedcba9876543210fedc")
    assert a != b
    # Until 2026-10-07 Path.with_suffix dropped the fingerprint: every fit of a
    # model shared <name>.nc, and a refit read back the fit it was replacing.
    assert a == (Path(tmp_path) / "rsa_l1.0123456789abcdef0123.nc", Path(tmp_path) / "rsa_l1.0123456789abcdef0123.json")

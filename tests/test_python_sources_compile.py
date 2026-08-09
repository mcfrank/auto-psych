"""Repository-wide syntax check for Python files outside pytest's import graph."""

from __future__ import annotations

from tests.paths import REPO_ROOT


SOURCE_ROOTS = ("src", "scripts", "analysis")


def test_all_python_sources_compile() -> None:
    """Every checked-in Python source must compile, including standalone CLIs."""
    failures: list[str] = []
    for root_name in SOURCE_ROOTS:
        for path in sorted((REPO_ROOT / root_name).rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")
            except (OSError, SyntaxError) as exc:
                relative = path.relative_to(REPO_ROOT)
                failures.append(f"{relative}: {type(exc).__name__}: {exc}")

    assert not failures, "Python source compilation failed:\n" + "\n".join(failures)

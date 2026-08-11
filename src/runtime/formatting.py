"""Formatting helpers shared by the rendered result summaries."""

from __future__ import annotations


def format_number(value: object, ndigits: int = 3) -> str:
    """Format a number to ``ndigits`` decimals, or ``n/a`` for missing values.

    Summary tables mix numbers with fields a run may not have produced (a
    single-model experiment has no runner-up, a single-repeat run has no ICC),
    so ``None`` renders as ``n/a`` instead of crashing the report. Anything
    non-numeric — a model name, a label — passes through as its string form.

    Note the sibling ``_fmt`` helpers in ``subjective_randomness/reporting.py``
    (4 significant digits) and ``discriminating_probe.py`` (``undefined`` for
    missing values) are deliberately *not* this function: their output strings
    are part of those reports' formats.
    """
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{ndigits}f}"
    return str(value)

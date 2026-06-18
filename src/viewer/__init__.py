"""Browser-based explorer for automated-psychology runs.

A live Flask server (:mod:`src.viewer.server`) scans ``data/outer_loop`` on
demand and exposes each run's artifacts — theory models, experiment design,
collected data, the inner model loop, and critiques — as JSON for a
single-page frontend. See :mod:`src.viewer.scan` for the directory-to-data
mapping and :mod:`src.viewer.models` for the payload schema.
"""

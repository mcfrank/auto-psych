"""Ingest external reference-game datasets into the canonical RSA trial schema.

Each source module (`mayn_demberg`, `sikos2021`) pins its public source files
by URL and sha256 (`fetch`), recodes them into rows of the column contract in
`common.COLUMNS` (the pragmods columns plus `source`, `messages` and
`covariates`), and records provenance. `run` is the CLI that fetches, derives
and writes; `combine` concatenates chosen sources (pragmods included) into one
responses CSV for the loop.

Where a derived CSV goes depends on the source's licence: a CC-BY source's
CSV is committed under the project's data directory, a source without a
licence is written to the gitignored cache (`data/rsa/external/`) and only its
pin file (URLs, hashes, the derived CSV's hash) is committed. The column
contract and every coding decision are in
`src/pipelines/outer_loop/projects/rsa_reference/data/README.md`.
"""

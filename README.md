# Palisades Fire 2024 — Peak-Flow Analysis (WEPPcloud)

This repo contains scripts + a LaTeX report for analyzing **peak flow / return-interval behavior** in WEPPcloud run `upset-reckoning`, including:
- burned (base) vs undisturbed comparisons
- sub-daily channel-hydrograph “desynchronization” diagnostics
- homogeneous (omni) scenario controls (`uniform_moderate`, `uniform_high`)

Intermediate artifacts under `tmp_*` are intentionally kept.

## Python environment (uv)

- Create/sync the virtualenv: `uv sync`
- Run scripts with the venv Python, e.g.: `.venv/bin/python skills/weppcloud-agent/scripts/desynchronization_analysis.py --help`

## Build the report

- Build: `latexmk -pdf -interaction=nonstopmode -halt-on-error report_upset_reckoning_hydroshape.tex`

## Handoff

See `AGENTS.md` for the full “where we left off” notes, key scripts, and reproduction commands.

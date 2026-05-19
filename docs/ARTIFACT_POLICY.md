# Artifact Policy

The repository tracks source files and compact metric summaries. Large
generated outputs remain outside version control.

## Tracked

- experiment scripts under `experiments/`
- documentation under `docs/`
- compact summary JSON files under `results/`
- chart proposal JSON files under `results/llm_chart_proposals/`

## Untracked

- videos and extracted frame sequences
- raw per-run simulator outputs
- model checkpoints and tensor dumps
- machine-specific configuration files

## Figure Promotion

When a generated image becomes part of a published result, record the command
needed to reproduce it in the relevant experiment notes.

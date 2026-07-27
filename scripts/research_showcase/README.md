# Research showcase toolkit

This directory contains read-only evaluation and reporting tools for turning
existing benchmark episodes into researcher-facing evidence. It is deliberately
separate from collection, environment, and schema code.

Source HDF5 and MP4 files are never modified. Generated inventories and visual
reports go to `docs/research_feature_showcase/` by default.

## Step 1: inventory

```bash
uv run python scripts/research_showcase/inventory.py \
  --config scripts/research_showcase/config/claims.json \
  --output docs/research_feature_showcase/evidence_inventory.json
```

The inventory records episode/video frame alignment, cameras, graph relations,
action intervals, object identities, provider provenance, and any missing fields.
Paths in the configuration are repository-relative, so the toolkit continues to
work if the repository root is renamed.

Later stages will consume the inventory to extract keyframes, render graph
snapshots/deltas, and assemble claim-specific reports.

## Step 2: unified-graph pilot

```bash
uv run python scripts/research_showcase/build_pilot.py
```

This creates a 16:9 pilot under `docs/research_feature_showcase/sauce_can_in_cabinet/` with the episode video, scene keyframes, focused graphs, graph deltas, paired panels, machine-readable evidence, and a narrative report.

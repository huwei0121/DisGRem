# Public DisGRem Agent Instructions

## Scope and Inheritance

- This file governs this project. Read the relevant README and applicable
  subdirectory `AGENTS.md` files before editing.
- In the standard `2026Projects/CATEGORY/PROJECT` layout, explicitly read
  `../../AGENTS.md`, `../AGENTS.md`, and relevant `../../CONVENTIONS.md` sections.
  They may sit outside automatic discovery at this independent Git root.
- For a relocated checkout/worktree, locate the actual workspace first;
  do not assume these relative counterpart paths still resolve.
- In a standalone clone, this file and the README remain the local contract.
  Ordinary setup/tests must not require private workspace files or siblings.
- Paths below are project-relative unless explicitly marked otherwise.
  Local facts and exceptions specialize shared defaults while retaining
  research integrity, raw evidence, privacy, and independent Git boundaries.
- Follow the user's current enabled-skill policy. Use relevant enabled skills
  without a mandatory pipeline or automatic reactivation.

## Project Identity

This is the public MIT-licensed implementation of DisGRem, CeDisGRem, and
AdaDisGRem for decentralized consensus optimization. It is tied to the paper
but should remain a clean public code release.

## Key Paths

- Command-line entry: `codes/main.py`.
- Proposed methods: `codes/solvers/disgrem/`.
- Baselines: `codes/solvers/baselines/`.
- Problems and data interfaces: `codes/problems/`.
- Benchmarks: `codes/experiments/benchmarks/`.
- Ablations: `codes/experiments/ablation/`.
- Tests: `codes/tests/`.

## Organization and Versioning

- Retain `codes/main.py` as the CLI, with solver families, baselines,
  problems, experiment drivers, and tests under their existing `codes/` paths.
- Preserve runtime/development/locked dependency files and public registry
  names. Git history records working changes; release versions identify API changes.
- Local `codes/results/` outputs and reviewer artifacts are not part of the
  code-only source release. Their separate freeze/reproduction contract remains
  documented in the README and artifact tooling.

## Public Release Rules

- Do not add private manuscript drafts, review material, or unpublished result
  dumps.
- Preserve the distinction between DisGRem, communication-efficient variants,
  adaptive variants, and baselines.
- Keep generated outputs under `codes/results/` out of commits unless the user
  explicitly promotes them for release.
- Maintain clean setup and test commands for outside users.

## Verification

Start each command block independently at the project root unless another
working directory is stated. Run checks relevant to the changed artifact;
instruction-only edits need reference and diff checks, not full experiments.

```powershell
cd codes
python -m pytest tests
python main.py regular
```

Use targeted benchmark or ablation scripts when changing solver behavior,
communication schedules, graph generation, or adaptive regularization.

## Research Rules

- Claims about convergence, communication efficiency, and scalability must be
  traceable to the paper or reproducible experiment scripts.
- Do not weaken graph, smoothness, or consensus assumptions in prose or comments.

## Git Policy

- Inspect this repository/worktree's status and diff before staging named
  task-related files; preserve unrelated user changes and sibling Git boundaries.
- Commit this project's changes separately. Push only within the user's stated
  scope after checking the actual remote, branch, visibility, and outgoing commits.
- Do not publish private sources/results, rewrite history, or delete evidence
  as a side effect of routine organization.

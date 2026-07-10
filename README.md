# DisGRem

Reference implementation for the Distributed Gradient-Regularized Newton
Method (DisGRem) and its communication-efficient and adaptive variants.

**Status:** preprint / SIOPT under review

**Paper:** [arXiv:2605.19396](https://arxiv.org/abs/2605.19396)

**License:** MIT

## Overview

DisGRem is a decentralized second-order method for consensus optimization over
networks. Each agent solves a local regularized Newton system with vanishing
gradient-norm regularization and communicates through scheduled gossip mixing.
The implementation includes the full DisGRem method, the communication-efficient
CeDisGRem variant, and the adaptive AdaDisGRem variant used in the paper.

## Repository Structure

```text
codes/
  main.py                  unified command-line entry point
  solvers/
    disgrem/               proposed DisGRem-family methods
    baselines/             comparison algorithms
  problems/                benchmark objectives and data interfaces
  experiments/
    benchmarks/            main benchmark and scalability experiments
    ablation/              robustness, communication, and adaptive studies
  scripts/                 figure and table regeneration helpers
  utils/                   graph generation, logging, plotting, and exports
  tests/                   smoke and diagnostic tests
```

Generated outputs are written under `codes/results/` and are not part of the
source release. A separate, hash-verified reviewer artifact freezes the exact
raw outputs used for manuscript verification.

## Setup

Requires Python 3.10 or newer.

```bash
cd codes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development and the public test gate, install `requirements-dev.txt`
instead; it includes the runtime requirements and pytest.

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

`requirements-lock.txt` records the complete Python 3.12 environment used for
the frozen reviewer artifact, including transitive dependencies.

```bash
pip install -r requirements-lock.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\activate
```

## Running Experiments

All commands below are run from `codes/`.

```bash
python main.py regular
python main.py robust
python main.py comm
python main.py ada
python main.py scale
python main.py tune
```

To run the full suite used in the paper:

```bash
python main.py all
python scripts/replot.py
```

The full benchmark can take many hours on a modern multi-core CPU. Individual
experiment modes can be run independently. The tuning mode uses disjoint seed
namespaces for parameter selection and held-out evaluation and resumes only from
source- and configuration-matched caches.

For a deliberately reduced smoke run, `LOG_SCHEDULE_MAXIT` overrides every
per-objective iteration budget. Such a run is recorded as non-paper-grade and
is rejected by the artifact freeze gate.

## Reproducibility Contract

The experiments use connected random geometric graphs: ten nodes are sampled
uniformly in the two-dimensional unit square and connected when their distance
is below `0.5`. The starting point and graph are regenerated for each Monte
Carlo replicate and paired across algorithms.

Every mode writes a run manifest containing the source-tree hash, Git state,
runtime environment, command, seeds, graph model, and the explicit boundary
between the analyzed algorithm and the capped experimental implementation.
Each experiment also exports machine-readable raw trajectories as compressed
JSON. After all modes complete, freeze and verify the artifact with:

```bash
python scripts/freeze_artifact.py freeze
python scripts/freeze_artifact.py verify
```

The freeze gate rejects incomplete or reduced protocols, dirty source runs,
changed source hashes, missing raw files, malformed gzip/JSON, incorrect run
counts, and checksum mismatches.

## Algorithms

| Registry callable | Paper name | Role |
|------|------------|------|
| `utils/alg/alg_bank.py::disgrem` | DisGRem | Full Hessian exchange |
| `utils/alg/alg_bank.py::ce_disgrem_adaptive` | CeDisGRem | Communication-efficient exchange |
| `utils/alg/alg_bank.py::ada_disgrem` | AdaDisGRem | Adaptive regularization |
| `utils/alg/alg_bank.py::ce_ada_disgrem_adaptive` | CeAdaDisGRem | Adaptive and communication-efficient |
| `solvers/disgrem/dis_greqm.py::dis_greqm` | DisGreQm | Quasi-Newton variant |

Baselines include EXTRA, DIGing, DQM, ESOM, SONATA, NetworkGIANT, and DisQN.

## Citation

```bibtex
@article{hu2026disgrem,
  title   = {Distributed Gradient-Regularized Newton Method:
             Scheduled Consensus and {$\mathcal{O}(\varepsilon^{-1})$}
             Global Iteration Complexity},
  author  = {Hu, Wei and Xie, Pengcheng and Yuan, Ya-Xiang and Zhang, Li},
  journal = {arXiv preprint arXiv:2605.19396},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE`.

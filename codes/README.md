# DisGRem Code

See the [project README](../README.md) for full documentation, installation, and usage instructions.

## Quick Reference

```bash
cd codes
python main.py [mode] [arg]
```

| Mode | Description |
|------|-------------|
| `regular` | Main 9-function benchmark (d=30, 20 MC) |
| `robust` | Robustness study (100 MC + param sweep) |
| `comm` | Communication cost study |
| `ada` | Adaptive mechanism study |
| `scale` | Dimension scalability study |
| `tune` | Seed-separated nested tuning and held-out evaluation |
| `all` | Run regular + comm + ada + robust + tune + scale sequentially |
| `clean` | Remove all generated results and caches |

Each mode writes an auditable run manifest under `results/run_manifests/` and
raw compressed JSON under its `data_log/` directory. Run
`python scripts/freeze_artifact.py freeze` only after all six modes complete.

# Reproducibility status

## Checks performed for publication

- Parsed all 14 Python files without importing or executing research code.
- Verified published Python and environment files match the archive byte for byte.
- Scanned included text for known personal details, machine paths, contact information, and credential patterns.
- Rendered four figures to PNG and inspected them visually.
- Started a new repository without copying private Git history.

## Missing or omitted components

| Component | Consequence |
| --- | --- |
| RF measurements and metadata dictionaries | Dataset-backed training cannot be reproduced from this repository alone. |
| utilities_tsai.py | Contributed dataset loader withheld pending redistribution permission. |
| Original envs and utils package layout | Imports require restoration or deliberate refactoring. |
| utils.logger_utils and some registration modules | Additional runtime dependencies need recovery. |
| Evaluation scripts and complete experiment configuration | Thesis metrics cannot be independently regenerated from this snapshot. |
| Checkpoints and raw logs | Not included; no inference or log-derived rerun is offered. |

## Environment

environment.yml is an archived Linux Conda export. The private submission misleadingly named the same Conda YAML src/requirements.txt; only the accurately named copy is included. Its machine-specific prefix had already been removed in archive preparation.

The export contains Linux build pins and old dependency versions. It was not installed or validated for this release. Do not pass it to pip install -r or assume Windows compatibility.

## Restoration needed before training

Obtain authorized data and contributed utilities, restore missing modules and original package layout, review dependency compatibility, and validate observations, rewards, and agent/environment interfaces with small checks before attempting long training runs. These remain outstanding tasks.

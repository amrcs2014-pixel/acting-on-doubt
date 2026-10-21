# Acting on Doubt (AoD)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23543240.svg)](https://doi.org/10.5281/zenodo.23543240)

Code, simulator and results accompanying the manuscript *Acting on Doubt: Priced Escalation and Delay-Corrected Learning for Budget-Constrained IoT Intrusion Response*. The manuscript itself is not included in this repository.

The study is a **trace-driven simulation**:
- **Real flows:** every detector score comes from a real flow of a public IoT dataset.
- **Simulated dynamics:** the network, the attacker, energy and the label-delay process are simulated.
- **No hardware:** nothing was measured on hardware. Energy is modelled from a published per-inference figure and swept.

## Layout

| Path | What it is |
|---|---|
| `data/` | Metadata of the CICIoT2023, Edge-IIoTset and NF-ToN-IoT-v3 subsamples. The datasets themselves are not redistributed; see `data/README.md` and `scripts/prep_data.py` |
| `results/cache/` | Cached detector score pools and DP threshold tables (enough to rerun every experiment) |
| `aod/detectors.py` | Device (8-feature, calibrated logistic) and gateway (calibrated boosted trees) detectors; score pools; propagation-phase attack families |
| `aod/sim.py` | Vectorised network simulator: belief filter, escalation, gateway capacity, token bucket, coordinator, label channel, IPCW weights |
| `aod/dp.py` | Exact single-device DP (Theorem 2), structure checks and policy evaluation |
| `aod/policies.py` | AoD threshold policy, DP-table policy, and the baselines |
| `aod/train.py` | AoD-T trainer v3 (`train_aodt_v3`: corrected-label grid warm start, primal–dual ES refinement, held-out selection); batched grid tuning for baselines (with top-10 re-scoring) |
| `aod/ppo.py` | AoD-N: Lagrangian PPO with belief shaping; deep-RL baselines |
| `aod/exp.py` | Resumable multi-process runner for all experiments; `is_current` hides superseded records |
| `aod/analysis.py`, `aod/stats.py` | Figures, tables, IQM, two-level (seed × network) bootstrap, Mann–Whitney U + Holm |
| `results/raw/*.jsonl` | One JSON record per (experiment, dataset, method, seed, setting) |
| `scripts/` | Unattended runs (`watchdog.ps1` + `results/jobs.txt`, `bg.py`), diagnostics, the E3 structure analysis |
| `tests/` | Simulator consistency checks; batched-tuning equivalence |

Records carry a version tag. AoD-T records have `trainer="v3"`; baseline and oracle-tuned records have `grid="v3"`. Older records stay in `results/raw` for the record (improvement loop, see `PROGRESS.md`), but `aod.exp.is_current` keeps them out of every table.

## Protocol

- **Splits.** Detectors are fit on the train split. The filter's likelihood tables come from the calibration split.
- **Training and tuning.** Every method is trained or tuned on networks driven by calibration-split flows.
- **Evaluation.** Every reported number comes from the same 32 held-out test networks, driven by test-split flows.
- **Baselines.** They are tuned with full ground truth. AoD-T learns only from delayed, censored confirmations.

## Reproduce

The detector score pools and DP tables are cached in `results/cache/`, so every step below runs **without the source datasets**. To retrain the detectors from the original data instead, see [`data/README.md`](data/README.md).

```bash
pip install numpy scipy pandas scikit-learn matplotlib torch pyarrow
python -m tests.test_consistency         # simulator sanity checks
python -m tests.test_batched_tuning      # batched tuning == one-point-at-a-time tuning
python -m aod.exp E0 --workers 8         # exact-DP structure grid (810 configurations)
python -m aod.exp E1 --workers 5 --skip task_aodt                                   # baselines + AoD-DP
python -m aod.exp E1 --workers 3 --skip task_baselines,task_aoddp --out E1aod       # AoD-T (trainer v3)
python -m aod.exp E1b --workers 2        # oracle-tuned two-threshold reference
python -m aod.exp E2 --workers 6         # delayed / censored labels
python -m aod.exp E3 --workers 2         # neural variants (GPU optional)
python -m scripts.e3_structure           # does AoD-N learn the two-threshold structure?
python -m aod.exp E4 --workers 6         # robustness (evaluation-time shifts)
python -m aod.exp E4u --workers 3        # unseen attack families
python -m aod.exp E4lat --workers 3      # verdict latency
python -m aod.exp E5 --workers 6         # scale and coordinator ablations
python -m aod.exp E1net --workers 3      # per-network costs for the main-table statistics
python -m aod.analysis                   # figures/ and LaTeX/JSON tables (written to paper/tables/)
python scripts/make_diag_table.py        # trainer diagnostics table (appendix)
```

E4 and E5 re-evaluate the E1/E1aod/E1b policies, so run them after those. On this laptop, unattended runs should go through `scripts/watchdog.ps1`, which reads `results/jobs.txt` and restarts any job that dies. Launch it from a long-lived shell, not through WMI.

Software: Python 3.11, NumPy 2.4, SciPy 1.17, scikit-learn 1.6, PyTorch 2.14.

## Licence and citation

- **Code** (`aod/`, `scripts/`, `tests/`): MIT, see [`LICENSE`](LICENSE).
- **Figures and result records** (`figures/`, `results/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Citing:** see [`CITATION.cff`](CITATION.cff). The archived release v1.0.0 is at Zenodo, DOI [10.5281/zenodo.23543240](https://doi.org/10.5281/zenodo.23543240).

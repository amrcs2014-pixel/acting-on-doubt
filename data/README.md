# Data

The processed dataset files (`<name>.parquet`) are **not** in this repository. The three source datasets have their own licences and terms. The `<name>.json` files here describe the exact subsamples we used: features, families, class counts, number of groups and how the subsample was drawn.

## You do not need the datasets to rerun the experiments

`results/cache/pool_*.npz` holds the detector score pools that the simulator uses: per-flow score bins of the device and gateway detectors on the calibration and test splits, plus the likelihood histograms. Every experiment (`python -m aod.exp ...`) and every table (`python -m aod.analysis`) runs from these caches.

## Rebuilding the data from the originals

You need this only to retrain the detectors.

1. Download the originals into one folder `RAW`:

   | Dataset | Source | Expected path under `RAW` |
   |---|---|---|
   | CICIoT2023 | Canadian Institute for Cybersecurity, <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | `ciciot2023/CSV/CSV/<attack type>/*.csv` |
   | Edge-IIoTset | Ferrag et al., <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot> | `edgeiiot/DNN-EdgeIIoT-dataset.csv.zip` |
   | NF-ToN-IoT-v3 | University of Queensland NetFlow v3 datasets, <https://staff.itee.uq.edu.au/marius/NIDS_datasets/> | `netflow/NF-ToN-IoT-v3.zip` |

2. Build the subsamples:

   ```bash
   python scripts/prep_data.py RAW
   ```

3. Retrain the detectors, replacing the cached pools:

   ```bash
   python -c "from aod.detectors import build_pool; [build_pool(n, force=True) for n in ('ciciot2023','edgeiiot','nfton_v3')]"
   ```

Please cite the dataset papers: CICIoT2023 (Neto et al., Sensors 2023), Edge-IIoTset (Ferrag et al., IEEE Access 2022), ToN-IoT (Moustafa, 2021) and the NetFlow v3 datasets (Luay et al., arXiv:2503.04404). Full references are in `paper/refs.bib`.

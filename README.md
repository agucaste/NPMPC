This is the repository for the paper ["Data-driven Acceleration of MPC with guarantees"](https://arxiv.org/pdf/2511.13588). 

## Installation

- **Conda (recommended):** create the environment from the provided recipe:

```bash
conda env create -f conda-recipe.yaml
conda activate npmpc
```

Adjust the environment name if your conda recipe defines a different one.

## Project structure

- **core/nn_policy.py**: implements the `NNRegressor` (1 nearest-neighbor regressor) and `NNPolicy` (greedy policy w.r.t. the upper bound).
- **config.yaml**: configuration for each environment and default algorithm parameters.
- **core/evaluator.py**: run the uniform-sampling data collection and evaluation pipeline.
- **core/smart_sampling_evaluator.py**: implements the verification / smart-sampling procedure (Algorithm 2 in the paper).
- **core/load_and_eval.py**: utilities to load stored runs and evaluate batches of saved datasets.
- **results/**: experiments are saved under `results/<env_name>/<date>/...`.

## How to run

There are two main ways to collect data and evaluate:

1. Uniform sampling (baseline)

```bash
python -m core.evaluator
```

2. Smart sampling (Algorithm 2 from the paper)

```bash
python -m core.smart_sampling_evaluator
```

Notes:
- Both entry scripts rely on the environment configuration in `config.yaml` and construct the model/controllers via the code in `core/constructor.py`.
- If you want a specific environment, set the environment identifier in the script.

## Stored runs and evaluation

- Collected runs are saved under `results/<env_name>/<date>/datasets`.
- To evaluate a set of saved runs (batch evaluation), use `core/load_and_eval.py` which loads the saved datasets and computes aggregate metrics.

## Quick references

- [core/nn_policy.py](core/nn_policy.py)
- [config.yaml](config.yaml)
- [core/evaluator.py](core/evaluator.py)
- [core/smart_sampling_evaluator.py](core/smart_sampling_evaluator.py)
- [core/load_and_eval.py](core/load_and_eval.py)
- [results](results)

## Contact
Contact: [acaste11@jhu.edu](mailto:acaste11@jhu.edu)

### Citation
If you use this code, please cite:
```bibtex
@article{castellano2025data,
	title={Data-driven Acceleration of MPC with Guarantees},
	author={Castellano, Agustin and Pan, Shijie and Mallada, Enrique},
	journal={arXiv preprint arXiv:2511.13588},
	year={2025}
}
```
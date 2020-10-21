# Numerically Solving Parametric Families of High-Dimensional Kolmogorov Partial Differential Equations via Deep Learning
> Accompanying code for NeurIPS 2020 paper.
> Deep Learning based algorithm for solving a parametrized family of high-dimensional Kolmogorov PDEs.
> Implemented in PyTorch and Tune.

![Illustration of the algorithm](/figures/algorithm.png)

## Reproducing the Experiments

To run the experiments and visualize the results open the jupyter notebook `experiments.ipynb`.
For reproducibility we recommend to use the docker container defined by `Dockerfile` (see [Docker Tutorial](https://docs.docker.com/get-started/part2/)).

### Our setup:

- DGX-1 server
- Ubuntu 18.04.3, Python 3.6.9, Torch 1.5 (as given by the [NVIDIA-Docker](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with base image `nvcr.io/nvidia/pytorch:20.03-py3`)
- additional requirements as specified by `requirements.txt`

### Run experiments:

| Experiment (and reference in the paper)                                    | Command (adapt `--gpus` if necessary)                |
|----------------------------------------------------------------------------|------------------------------------------------------|
| Black-Scholes model (Table 1, Fig. 3,4,5,6,7,8)                            | `python main.py --mode=avg_bs --gpus=2`              |
| Heat-equation with paraboloid initial condition (Table 3)                  | `python main.py --mode=avg_heat_paraboloid --gpus=2` |
| Heat-equation with Gaussian initial condition (Table 4)                    | `python main.py --mode=avg_heat_gaussian --gpus=2`   |
| Basket put option (Table 2)                                                | `python main.py --mode=avg_basket --gpus=4`          |
| Cost vs. input dimension (Fig. 9)                                          | `python main.py --mode=dims_heat_paraboloid --gpus=2`|
| Ablation study Black-Scholes model (Table 7)                               | `python main.py --mode=compare_nets_bs --gpus=2`     |
| Ablation study heat-equation (Table 8)                                     | `python main.py --mode=compare_nets_heat --gpus=2`   |
| Hyperparameter search (Table 6)                                            | `python main.py --mode=optimize_bs --gpus=2`         |

### Visualize results:

1. **Jupyter notebook:** Open the notebook `experiments.ipynb` and run section `Analyze experiments`
2. **Tensorboard:** Run `tensorboard --logdir exp` or `tensorboard --logdir exp/experiment_xyz`


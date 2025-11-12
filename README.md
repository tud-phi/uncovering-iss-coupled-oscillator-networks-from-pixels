# Input-to-State Stable Coupled Oscillator Networks for Closed-form Model-based Control in Latent Space

This repository contains the code for the paper **Input-to-State Stable Coupled Oscillator Networks for Closed-form Model-based Control in Latent Space**
by Maximilian Stölzle and Cosimo Della Santina, which will be presented at **NeurIPS 2024**.
In particular, this codebase includes:

- A JAX implementation of the Coupled Oscillator Network (CON) the CFA-CON models.
- The training code for learning latent-space dynamics of mechanical systems with CONs, CFA-CONs, and other baselines (e.g., NODE, RNN, GRU, coRNN, etc.)
- Model-based latent-space controllers.
- Analysis and visualization scripts for generating the figures in the paper.

**Abstract:** 
Even though a variety of methods (e.g., RL, MPC, LQR) have been proposed in the literature, efficient and effective latent-space control of physical systems remains an open challenge.
A promising avenue would be to leverage powerful and well-understood closed-form strategies from control theory literature in combination with learned dynamics, such as potential-energy shaping.
We identify three fundamental shortcomings in existing latent-space models that have so far prevented this powerful combination: (i) they lack the mathematical structure of a physical system, (ii) they do not inherently conserve the stability properties of the real systems. Furthermore, (iii) these methods do not have an invertible mapping between input and latent-space forcing.
This work proposes a novel Coupled Oscillator Network (CON) model that simultaneously tackles all these issues. 
More specifically, (i) we show analytically that CON is a Lagrangian system - i.e., it presses well-defined potential and kinetic energy terms. Then, (ii) we provide formal proof of global Input-to-State stability using Lyapunov arguments.
Moving to the experimental side, (iii) we demonstrate that CON reaches SoA performance when learning complex nonlinear dynamics of mechanical systems directly from images.
An additional methodological innovation contributing to achieving this third goal is an approximated closed-form solution for efficient integration of network dynamics, which eases efficient training.
We tackle (iv) by approximating the forcing-to-input mapping with a decoder that is trained to reconstruct the input based on the encoded latent space force.
Finally, we leverage these four properties and show that they enable latent-space control. We use an integral-saturated PID with potential force compensation and demonstrate high-quality performance on a soft robot using raw pixels as the only feedback information.

## Citation

This simulator is part of the publication _Input-to-State Stable Coupled Oscillator Networks for Closed-form Model-based Control in Latent Space_ presented at as a **Spotlight paper** at the _Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)_ in Vancouver, Canada. 
You can find the publication online on arXiv: https://arxiv.org/abs/2409.08439

Please use the following citation if you use our software in your (scientific) work:

```bibtex
@inproceedings{stolzle2024input,
  title={Input-to-State Stable Coupled Oscillator Networks for Closed-form Model-based Control in Latent Space},
  author={St{\"o}lzle, Maximilian and Della Santina, Cosimo},
  booktitle={Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024},
}
```

## Installation

### Install the system dependencies

On Ubuntu, please install the following system dependencies:

```bash
sudo apt install ffmpeg
```

or with Conda:

```bash
conda install -c conda-forge ffmpeg
```

### Install the Python dependencies

This library requires Python 3.10 or higher. Please install the Python dependencies using the following command:

```bash
pip install -r requirements.txt
```

#### Using `uv` 

You can also set up a Python virtual environment using [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager, with the following commands:

```bash
uv venv
source .venv/bin/activate
```

After this, you should see the environment activated in the prompt of your terminal.:
```bash
(uncovering-iss-coupled-oscillator-networks-from-pixels) :/$ which python
/path/to/uncovering-iss-coupled-oscillator-networks-from-pixels/.venv/bin/python 
```
Finally, install the dependencies with `uv pip`:

```bash
uv pip install -r requirements.txt
```

Remember to `source .venv/bin/activate` in any new terminal before executing the scripts.

## Usage

### Add the project to your PYTHONPATH

On Linux systems, we need to add the directory of the project to the `PYTHONPATH` environment variable. 
This can be done by running the following command:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

Alternatively, you can also use our helper script:

```bash
source ./01-configure-env-vars.sh
```

Afterwards, you can run the scripts in the `examples` folder. Some examples are given below.

## Generating the datasets

The compressed Tensorflow dataset can be generated using the following commands:

### Single Pendulum

```bash
tfds build datasets/pendulum --data_dir data/tensorflow_datasets --config single_pendulum_32x32px_h-101 --overwrite
```

### Planar PCS robot

#### One Constant Strain segment

```bash
tfds build datasets/planar_pcs --data_dir data/tensorflow_datasets --config cs_32x32px_h-101 --overwrite
```

#### Two Piecewise Constant Curvature segments

```bash
tfds build datasets/planar_pcs --data_dir data/tensorflow_datasets --config pcc_ns-2_32x32px_h-101 --overwrite
```

#### Three Piecewise Constant Curvature segments

```bash
tfds build datasets/planar_pcs --data_dir data/tensorflow_datasets --config pcc_ns-3_32x32px_h-101 --overwrite
```

### Reaction-Diffusion

```bash
tfds build datasets/reaction_diffusion --data_dir data/tensorflow_datasets --config reaction_diffusion_default --overwrite
```

### N-Body problem

```bash
tfds build datasets/nbody_problem --data_dir data/tensorflow_datasets --config nb-2_h-101_32x32px --overwrite
```

## Analysis of the approximate closed form solution

The following command can be used to analyze the behavior and performance of the approximate closed form solution (e.g., measuring the error w.r.t. to the CON dynamics, plotting the approximated solution, benchmarking the computation time, etc.):

```bash
python examples/cfa_con/evaluate_cfa_con.py
```

## Tuning of the hyperparameters

The hyperparameters of the models can be tuned using the following command:

```bash
python examples/tuning/tune_planar_pcs_dynamics_autoencoder.py
```

Subsequently, navigate into the log folder of the experiment that you just started and run the following command to visualize the results:

```bash
optuna-dashboard sqliet:///optuna_study.db
```

## Sweep across the latent dimensions and seeds

We can sweep across the latent dimensions and seeds. For each combination of latent dimension and see, we train the model and evaluate it on the test set. The following command can be used to run the sweep:

```bash
python examples/sweep/sweep_planar_pcs_dynamics_autoencoder.py
```

The script will store the results inside the `sweep_results.dill` dictionary in the log folder of the experiment.
Subsequently, statistics across seeds can be computed using the following command:

```bash
python examples/sweep/analyze_sweep.py
```

## Latent-space control

The experiment of exploiting the coupled oscillator network for latent-space control can be run using the following command:

```bash
python examples/control/control_planar_pcs_dynamics_autoencoder_setpoint_sequence.py
```

## Tips & Tricks

### GPU memory allocation

If your GPU runs out of memory immediately after launching a JAX script, for example with the error:

```
INTERNAL: RET_CHECK failure (external/org_tensorflow/tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:626) dnn != nullptr 
```

please reduce as documented [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) the amount of memory 
pre-allocated to the GPU.

## Determinism

**Attention:** XLA on GPU is not deterministic by default even when setting seeds for random number generation as documented [here](https://github.com/google/jax/issues/13672). Therefore it is essential, to set the environment variable `XLA_FLAGS` to `--xla_gpu_deterministic_ops=true` to ensure determinism. This is done automatically when running `source ./01-configure-env-vars.sh`.

# Leveraging Input-to-State Stable Coupled Oscillator Networks for Learning Control-oriented Latent Dynamics from Pixels

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

## Usage

### Add the project to your PYTHONPATH

On Linux systems, we need to add the `src` folder to the `PYTHONPATH` environment variable. 
This can be done by running the following command:

```bash
export PYTHONPATH="${PYTHONPATH}://src"
```

Alternatively, you can also use our helper script:

```bash
source ./01-configure-env-vars.sh
```

Afterwards, you can run the scripts in the `src` folder. Some examples are given below.

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

#### Four Piecewise Constant Curvature segments

```bash
tfds build datasets/planar_pcs --data_dir data/tensorflow_datasets --config pcc_ns-4_32x32px_h-101 --overwrite
```

### N-Body problem

```bash
tfds build datasets/nbody_problem --data_dir data/tensorflow_datasets --config nb-2_h-101_32x32px --overwrite
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

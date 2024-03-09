# Uncovering Archetype Networks

## Installation

### Install System Dependencies
On Ubuntu, please install the following system dependencies:

```bash
sudo apt install ffmpeg
```

or with Conda:

```bash
conda install -c conda-forge ffmpeg
```

### Install the Python dependencies

Please install the Python dependencies using the following command:
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

## Generating the Tensorflow Dataset

The compressed Tensorflow dataset can be generated using the following commands:

### Single Pendulum

```bash
tfds build datasets/pendulum --data_dir data/tensorflow_datasets --config single_pendulum_64x64px --overwrite
```

### Planar PCS robot

```bash
tfds build datasets/planar_pcs --data_dir data/tensorflow_datasets --config cc_64x64px --overwrite
```

### N-Body problem

```bash
tfds build datasets/nbody_problem --data_dir data/tensorflow_datasets --config nb-2_h-101_32x32px --overwrite
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

# Learning Representations from first-principle Dynamics

## Installation

Please install the Python dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Add the project to your PYTHONPATH

On Linux systems, we need to add the `src` folder to the `PYTHONPATH` environment variable. 
This can be done by running the following command:

```bash
export PYTHONPATH="${PYTHONPATH}://src"
```

## Generating the Tensorflow Dataset

The compressed Tensorflow dataset can be generated from a raw dataset using the following command:
```bash
tfds build datasets/mechanical_system --data_dir data/tensorflow_datasets --overwrite
```

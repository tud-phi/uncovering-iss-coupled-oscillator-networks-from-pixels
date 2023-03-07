from jax import random

from src.training.load_dataset import load_datasets

if __name__ == "__main__":
    load_datasets("mechanical_system/single_pendulum")

from jax import random

from src.training.load_dataset import load_dataset

if __name__ == "__main__":
    datasets = load_dataset(
        "mechanical_system/single_pendulum",
        batch_size=32,
        normalize=True,
        grayscale=True,
    )
    print("train_ds: ", datasets["train"])
    print("val_ds: ", datasets["val"])
    print("test_ds: ", datasets["test"])

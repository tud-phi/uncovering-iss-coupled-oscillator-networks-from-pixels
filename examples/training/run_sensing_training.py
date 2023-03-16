from jax import random
import tensorflow as tf

from src.neural_networks.simple_cnn import Encoder
from src.training.tasks import sensing
from src.training.load_dataset import load_dataset
from src.training.loops import run_training

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

num_epochs = 25
batch_size = 8
base_lr = 5e-4
warmup_epochs = 2

if __name__ == "__main__":
    datasets = load_dataset(
        "mechanical_system/single_pendulum_64x64px",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # initialize the model
    nn_model = Encoder(latent_dim=1, img_shape=(64, 64, 1))

    # call the factory function for the sensing task
    task_callables = sensing.task_factory(nn_model)

    # run the training loop
    (
        state,
        history,
    ) = run_training(
        rng=rng,
        train_ds=train_ds,
        val_ds=val_ds,
        nn_model=nn_model,
        task_callables=task_callables,
        num_epochs=num_epochs,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=0.0,
    )

    print("Final validation metrics:\n", val_metrics_history[-1])

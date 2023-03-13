from jax import random
import tensorflow as tf

from src.neural_networks.simple_cnn import Encoder
from src.training.tasks import sensing
from src.training.initialization import initialize_train_state
from src.training.load_dataset import load_dataset
from src.training.loops import train_epoch, eval_model
from src.training.optim import create_learning_rate_fn

# prevent tensorflow from loading everything onto the GPU, as we don't have enough memory for that
tf.config.experimental.set_visible_devices([], "GPU")

# initialize the pseudo-random number generator
seed = 0
rng = random.PRNGKey(seed=seed)

num_epochs = 1
batch_size = 32

if __name__ == "__main__":
    datasets = load_dataset(
        "mechanical_system/single_pendulum",
        seed=seed,
        batch_size=batch_size,
        normalize=True,
        grayscale=True,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    print("train_ds: ", train_ds)
    print("val_ds: ", val_ds)
    print("test_ds: ", test_ds)

    # create learning rate schedule
    lr_fn = create_learning_rate_fn(
        num_epochs,
        steps_per_epoch=len(train_ds),
        base_lr=1e-4,
        warmup_epochs=0,
    )

    # initialize the model
    nn_model = Encoder(latent_dim=1)

    # call the factory function for the sensing task
    task_callables = sensing.task_factory(nn_model)

    # extract dummy batch from dataset
    nn_dummy_batch = next(train_ds.as_numpy_iterator())
    # assemble input for dummy batch
    nn_dummy_input = task_callables.assemble_input_fn(nn_dummy_batch)

    # initialize the train state
    state = initialize_train_state(
        rng, nn_model, nn_dummy_input=nn_dummy_input, learning_rate_fn=lr_fn
    )

    state, train_loss, epoch_metrics = train_epoch(
        0, state, train_ds, task_callables, lr_fn
    )
    print("training results:", epoch_metrics)

    val_loss, val_metrics = eval_model(state, val_ds, task_callables)
    print("validation results:", val_metrics)

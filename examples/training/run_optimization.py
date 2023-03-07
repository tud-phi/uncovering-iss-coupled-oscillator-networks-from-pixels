from flax.training.train_state import TrainState
import jax
from jax import random
import jax.numpy as jnp
import optax

from src.neural_networks.simple_cnn import Autoencoder

rng = random.PRNGKey(0)

if __name__ == "__main__":
    nn = Autoencoder(latent_dim=2)

    img_bt = jnp.ones((5, 32, 32, 1))
    nn_params = nn.init(rng, img_bt)["params"]

    # initialize the Adam with weight decay optimizer for both neural networks
    tx = optax.adamw(0.01, weight_decay=0.0)

    # create the TrainState object for both neural networks
    train_state = TrainState.create(
        apply_fn=nn.apply,
        params=nn_params,
        tx=tx,
    )

    z_bt = nn.apply({"params": nn_params}, img_bt, method=nn.encode)
    print("z_bt", z_bt.shape)

    loss_fn = lambda _nn_params: jnp.mean(nn.apply({"params": _nn_params}, img_bt, method=nn.encode) ** 2)

    loss_and_grad_fn = jax.value_and_grad(loss_fn, argnums=(0, ))
    loss, (grads, ) = loss_and_grad_fn(train_state.params)

    print("loss", loss)

    train_state = train_state.apply_gradients(
        grads=grads
    )

import jax.numpy as jnp

from src.losses.masked_mse import masked_mse_loss


def test_masked_mse_loss():
    input = jnp.array([[1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
    target = jnp.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, 1]])
    mse_loss = jnp.mean(jnp.square(input - target))
    print(f"mse_loss: {mse_loss}")
    masked_mse_loss = masked_mse_loss(input, target)
    print(f"masked_mse_loss: {masked_mse_loss}")

    assert masked_mse_loss > mse_loss

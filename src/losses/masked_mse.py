from jax import Array, debug, jit, lax
import jax.numpy as jnp


@jit
def masked_mse_loss(
        input: Array,
        target: Array,
        threshold_for_masking: float = 0.0,
        threshold_cond_sign: int = 1,
        masked_loss_weight: float = 0.5
) -> Array:
    """
    Compute the MSE loss separately for masked elements and unmasked elements.
    Arguments:
        input: the input array (for example, the predicted image)
        target: the target array (for example, the ground truth image)
        threshold_for_masking: the threshold for masking applied to the target array.
            if threshold_cond_sign * target(i) > threshold_for_masking, the element is masked.
        threshold_cond_sign: the sign of the threshold condition. If -1, the condition is target(i) < threshold_for_masking.
        masked_loss_weight: the weight for the masked loss. The weight for the unmasked loss is 1 - masked_loss_weight.
    Returns:
        loss: the total MSE loss computed separately for masked elements and unmasked elements
    """
    mask = threshold_cond_sign * target > threshold_for_masking
    num_total_elements = jnp.prod(jnp.array(target.shape))
    num_masked_elements = jnp.sum(mask)

    masked_input = jnp.where(mask, x=input, y=target)
    inv_masked_input = jnp.where(mask, x=target, y=input)

    # compute loss for the masked elements and unmasked elements separately
    mse_loss_masked_area = jnp.sum(jnp.square(masked_input - target)) / num_masked_elements
    mase_loss_inv_masked_area = jnp.sum(jnp.square(inv_masked_input - target)) / (num_total_elements - num_masked_elements)

    # compute the total loss
    loss = masked_loss_weight * mse_loss_masked_area + (1.0 - masked_loss_weight) * mase_loss_inv_masked_area
    return loss

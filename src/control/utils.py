from jax import Array
import jax.numpy as jnp


def compute_settling_time_on_setpoint_trajectory(ts: Array, ref_ts: Array, traj_ts: Array, threshold: float = 0.02):
    """
    Compute the settling time on a setpoint trajectory.
    Arguments:
        ts: the time steps. Shape: (num_timesteps,)
        ref_ts: the reference setpoint trajectory. Shape: (num_timesteps, state_dim)
        traj_ts: the actual trajectory. Shape: (num_timesteps, state+dim)
        threshold: the width of the settling bounds. Default: 0.02
    Returns:
        mean_settling_time: the average settling time. Shape: ()
    """
    error = jnp.abs(ref_ts - traj_ts)

    """
    ref_value = ref_ts[0]
    step_size = jnp.abs(error[0])
    total_settling_time = 0.0
    time_since_step = 0.0
    settled = False
    num_steps = 1
    t_prior = ts[0]
    for time_idx in range(ts.shape[0]):
        t = ts[time_idx]

        if jnp.any(step_size == 0.0):
            settled = True
            continue

        if jnp.any(ref_value != ref_ts[time_idx]):
            step_size = jnp.abs(ref_ts[time_idx] - ref_value)
            time_since_step = 0.0
            settled = False
            num_steps += 1

        if settled is False:
            time_since_step += (t - t_prior)
            if jnp.all(error[time_idx] / step_size <= threshold):
                total_settling_time += time_since_step
                settled = True

        ref_value = ref_ts[time_idx]
        t_prior = t
    """
    ref_value = ref_ts[0]
    step_size = jnp.abs(error[0])
    step_size_stps = [step_size]
    step_value_stps = [ref_value]
    step_time_stps = [ts[0]]
    for time_idx in range(ref_ts.shape[0]):
        if jnp.any(ref_value != ref_ts[time_idx]):
            step_size = jnp.abs(ref_ts[time_idx] - ref_value)
            ref_value = ref_ts[time_idx]
            step_size_stps.append(step_size)
            step_value_stps.append(ref_value)
            step_time_stps.append(ts[time_idx])

    step_size_stps = jnp.array(step_size_stps)
    step_value_stps = jnp.array(step_value_stps)
    step_time_stps = jnp.array(step_time_stps)

    total_settling_time = 0.0
    for step_idx in range(len(step_size_stps)):
        if step_idx < len(step_size_stps) - 1:
            step_selector = (ts >= step_time_stps[step_idx]) & (ts < step_time_stps[step_idx + 1])
        else:
            step_selector = ts >= step_time_stps[step_idx]

        norm_error = error[step_selector] / step_size_stps[step_idx]
        settled = jnp.where(jnp.all(norm_error <= threshold, axis=-1), True, False)
        settled = jnp.cumprod(settled[::-1])[::-1]

        if jnp.sum(settled) > 0.0:
            settling_time = ts[step_selector][jnp.argmax(settled, keepdims=True)[0]] - ts[step_selector][0]
        else:
            settling_time = ts[step_selector][-1] - ts[step_selector][0]
        total_settling_time += settling_time
        print(f"Step {step_idx}, settling time: {settling_time}, last norm_error: {norm_error[-1]}")

    print("total_settling_time: ", total_settling_time)

    mean_settling_time = total_settling_time / step_time_stps.shape[0]

    return mean_settling_time

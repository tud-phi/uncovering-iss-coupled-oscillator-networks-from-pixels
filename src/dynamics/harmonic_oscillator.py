import jax
import jax.numpy as jnp


def harmonic_oscillator_closed_form_dynamics(
    t: jax.Array,
    t0: jax.Array,
    y0: jax.Array,
    m: jax.Array,
    gamma: jax.Array,
    epsilon: jax.Array,
    f: jax.Array,
) -> jax.Array:
    """
    Closed-form solution of the harmonic oscillator.
    https://scholar.harvard.edu/files/schwartz/files/lecture1-oscillators-and-linearity.pdf
    Args:
        t: time
        t0: initial time
        y0: initial state
        m: mass
        gamma: stiffness
        epsilon: damping coefficient
        f: force
    Returns:
        y: oscillator state at time t
    """
    x0, v0 = jnp.split(y0, 2)

    # natural frequency
    omega_n = jnp.sqrt(gamma / m)
    # damping ratio
    zeta = epsilon / (2 * jnp.sqrt(m * gamma))

    # https://tttapa.github.io/Pages/Arduino/Audio-and-Signal-Processing/VU-Meters/Damped-Harmonic-Oscillator.html
    alpha = zeta * omega_n
    beta = omega_n * jnp.sqrt(1 - zeta**2)
    lambda1, lambda2 = -alpha + beta * 1j, -alpha - beta * 1j
    # constants for the closed-form solution
    """
    c1 = (-v0 + lambda2 * (x0 - f_ext / jnp.diag(K))) / (lambda2 - lambda1)
    c2 = (v0 - lambda1 * (x0 - f_ext / jnp.diag(K))) / (lambda2 - lambda1)
    ctilde1 = c1 + c2
    ctilde2 = (c1 - c2) * 1j
    """
    ctilde1 = x0 - f / gamma
    ctilde2 = (-2*v0 + (lambda1 + lambda2)*(x0 - f / gamma))/(lambda2-lambda1)*1j

    x = (
        ctilde1 * jnp.cos(beta * (t - t0)) + ctilde2 * jnp.sin(beta * (t - t0))
    ) * jnp.exp(-(alpha * (t - t0))) + f / gamma
    x_d = -(
        (ctilde1 * alpha - ctilde2 * beta) * jnp.cos(beta * (t - t0))
        + (ctilde1 * beta + ctilde2 * alpha) * jnp.sin(beta * (t - t0))
    ) * jnp.exp(-alpha * (t - t0))

    y = jnp.real(jnp.concatenate([x, x_d])).astype(jnp.float64)

    return y

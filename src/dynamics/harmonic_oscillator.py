import jax
import jax.numpy as jnp

from .utils import apply_eps_to_array


def harmonic_oscillator_closed_form_dynamics(
    t0: jax.Array,
    t1: jax.Array,
    y0: jax.Array,
    m: jax.Array,
    gamma: jax.Array,
    epsilon: jax.Array,
    f: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """
    Closed-form solution of N independent harmonic oscillator.
    https://scholar.harvard.edu/files/schwartz/files/lecture1-oscillators-and-linearity.pdf
    Args:
        t0: start time [s]
        t1: final time [s]
        y0: initial state [m, m/s] as array of shape (2*N, )
        m: mass [kg] as array of shape (N, )
        gamma: stiffness [N/m] as array of shape (N, )
        epsilon: damping coefficient [Ns/m] as array of shape (N, )
        f: force [N] as array of shape (N, )
        eps: small number to add to avoid singularities
    Returns:
        y1: final oscillator state [m, m/s] as array of shape (2*N, )
    """
    x0, v0 = jnp.split(y0, 2)

    # add small number to stiffness to avoid singularities
    gamma = apply_eps_to_array(gamma, eps=eps)

    # cast to complex numbers
    x0, v0 = x0.astype(complex), v0.astype(complex)
    m, gamma, epsilon, f = m.astype(complex), gamma.astype(complex), epsilon.astype(complex), f.astype(complex)

    # natural frequency
    omega_n = jnp.sqrt(gamma / m)
    # damping ratio
    zeta = epsilon / (2 * jnp.sqrt(m * gamma))

    # https://tttapa.github.io/Pages/Arduino/Audio-and-Signal-Processing/VU-Meters/Damped-Harmonic-Oscillator.html
    alpha = zeta * omega_n
    beta = omega_n * jnp.sqrt(1 - zeta**2)
    lambda1, lambda2 = -alpha + beta * 1j, -alpha - beta * 1j

    # when d = 2 * sqrt(m * k) => zeta = 1 => lambda1 = lambda2 => lambda2-lambda1 = 0, the system is critically damped
    # theoretically, the solution would be different. However, this case will rarely happen in practice
    # therefore, we will just try to prevent the division by zero
    lambda_diff = lambda2 - lambda1
    lambda_diff_epsed = apply_eps_to_array(lambda_diff, eps=eps)

    # constants for the closed-form solution
    """
    c1 = (-v0 + lambda2 * (x0 - f_ext / jnp.diag(K))) / lambda_diff_epsed
    c2 = (v0 - lambda1 * (x0 - f_ext / jnp.diag(K))) / lambda_diff_epsed
    ctilde1 = c1 + c2
    ctilde2 = (c1 - c2) * 1j
    """
    ctilde1 = x0 - f / gamma
    ctilde2 = (
        (-2 * v0 + (lambda1 + lambda2) * (x0 - f / gamma)) / lambda_diff_epsed * 1j
    )

    x = (
        ctilde1 * jnp.cos(beta * (t1 - t0)) + ctilde2 * jnp.sin(beta * (t1 - t0))
    ) * jnp.exp(-(alpha * (t1 - t0))) + f / gamma
    x_d = -(
        (ctilde1 * alpha - ctilde2 * beta) * jnp.cos(beta * (t1 - t0))
        + (ctilde1 * beta + ctilde2 * alpha) * jnp.sin(beta * (t1 - t0))
    ) * jnp.exp(-alpha * (t1 - t0))

    y1 = jnp.real(jnp.concatenate([x, x_d])).astype(y0.dtype)

    return y1

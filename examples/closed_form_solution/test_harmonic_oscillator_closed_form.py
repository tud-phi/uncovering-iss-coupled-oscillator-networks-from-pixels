import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from functools import partial
import jax.numpy as jnp
import matplotlib.pyplot as plt

# time steps
dt = 1e-4
ts = jnp.arange(0.0, 60.0, dt)

# parameters
m = 1.0  # mass
gamma = 0.1  # stiffness
epsilon = 0.05  # damping coefficient

# natural frequency
omega_n = jnp.sqrt(gamma / m)
# damping ratio
zeta = epsilon / (2 * jnp.sqrt(m * gamma))


def harmonic_oscillator_ode_fn(
    t: jax.Array,
    y: jax.Array,
    *args,
    m: jax.Array,
    gamma: jax.Array,
    epsilon: jax.Array,
) -> jax.Array:
    """
    Harmonic oscillator ODE.
    Args:
        t: time
        y: oscillator state
        m: mass
        gamma: stiffness
        epsilon: damping
    Returns:
        y_d: derivative of the oscillator state
    """
    x, x_d = jnp.split(y, 2)
    x_dd = m ** (-1) * (-gamma * x - epsilon * x_d)
    y_d = jnp.concatenate([x_d, x_dd])
    return y_d


def closed_form_harmonic_oscillator(
    t: jax.Array, y0: jax.Array, m: jax.Array, gamma: jax.Array, epsilon: jax.Array
) -> jax.Array:
    """
    Closed-form solution of the harmonic oscillator with underdamping.
    https://scholar.harvard.edu/files/schwartz/files/lecture1-oscillators-and-linearity.pdf
    Args:
        t: time
        y0: initial state
        m: mass
        gamma: stiffness
        epsilon: damping
    Returns:
        y: oscillator state at time t
    """
    x0, v0 = jnp.split(y0, 2)

    if epsilon == 0.0:
        x = x0 * jnp.cos(jnp.sqrt(gamma) * t / jnp.sqrt(m)) + jnp.sqrt(
            m
        ) * v0 * jnp.sin(jnp.sqrt(gamma) * t / jnp.sqrt(m)) / jnp.sqrt(gamma)
        x_d = -jnp.sqrt(gamma) * x0 * jnp.sin(
            jnp.sqrt(gamma) * t / jnp.sqrt(m)
        ) / jnp.sqrt(m) + v0 * jnp.cos(jnp.sqrt(gamma) * t / jnp.sqrt(m))
    elif epsilon < 2 * jnp.sqrt(m * gamma):
        print("Underdamped oscillator")
        # https://tttapa.github.io/Pages/Arduino/Audio-and-Signal-Processing/VU-Meters/Damped-Harmonic-Oscillator.html
        alpha = zeta * omega_n
        beta = omega_n * jnp.sqrt(1 - zeta**2)
        lambda1, lambda2 = -alpha + beta * 1j, -alpha - beta * 1j
        # constants for the closed-form solution
        c1 = (lambda2 * x0 - v0) / (lambda2 - lambda1)
        c2 = (v0 - lambda1 * x0) / (lambda2 - lambda1)
        ctilde1 = c1 + c2
        ctilde2 = (c1 - c2) * 1j

        x = jnp.exp(-alpha * t) * (
            ctilde1 * jnp.cos(beta * t) + ctilde2 * jnp.sin(beta * t)
        )
        x_d = -jnp.exp(-alpha * t) * (
            (ctilde1 * alpha - ctilde2 * beta) * jnp.cos(beta * t)
            + (ctilde1 * beta + ctilde2 * alpha) * jnp.sin(beta * t)
        )
    else:
        raise NotImplementedError

    y = jnp.stack([x, x_d])
    return y


# Define the harmonic oscillator ODE term
ode_term = ODETerm(
    partial(harmonic_oscillator_ode_fn, m=m, gamma=gamma, epsilon=epsilon),
)

# Solve the harmonic oscillator ODE
y0 = jnp.array([1.0, 0.0])
sol = diffeqsolve(
    ode_term,
    Tsit5(),
    t0=ts[0],
    t1=ts[-1],
    dt0=dt,
    y0=y0,
    saveat=SaveAt(ts=ts),
    max_steps=ts.shape[-1],
)
y_ts_numerical = sol.ys

# evaluate the closed-form solution
y_ts_closed_form = jax.vmap(
    partial(closed_form_harmonic_oscillator, y0=y0, m=m, gamma=gamma, epsilon=epsilon)
)(ts)

# plot the position
plt.plot(ts, y_ts_numerical[:, 0], label="Numerical solution")
plt.plot(ts, y_ts_closed_form[:, 0], label="Closed-form solution")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.grid()
plt.box()
plt.title("Harmonic oscillator position")
plt.show()

# plot the velocity
plt.plot(ts, y_ts_numerical[:, 1], label="Numerical solution")
plt.plot(ts, y_ts_closed_form[:, 1], label="Closed-form solution")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()
plt.grid()
plt.box()
plt.title("Harmonic oscillator velocity")
plt.show()

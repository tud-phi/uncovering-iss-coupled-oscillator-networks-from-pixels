import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from functools import partial
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from src.dynamics.utils import apply_eps_to_array

# time steps
dt = jnp.array(1e-4)
ts = jnp.arange(0.0, 60.0, dt)

# parameters
num_units = 2
m = 1.0 * jnp.ones((num_units,))  # mass
K = 0.1 * jnp.eye(num_units)  # stiffness matrix
D = 0.05 * jnp.eye(num_units)  # damping matrix
# D = jnp.diag(2*jnp.sqrt(m * jnp.diag(K)))  # damping matrix for critically damped oscillator
match num_units:
    case 1:
        W = 1.5e-1 * jnp.array([[1.0]])  # coupling matrix
        b = 1.5e-1 * jnp.array([-0.5])  # bias
        # W = jnp.array([[0.0]])
        # b = jnp.array([0.0])
        y0 = jnp.array([1.0, 0.0])
    case 2:
        K = 0.1 * jnp.array([[1.0, 0.25], [0.25, 1.0]])
        D = 0.05 * jnp.array([[1.0, 0.3], [0.3, 1.0]])
        W = 2e-1 * jnp.array([[1.0, 0.5], [0.5, 1.0]])
        print("Eigvals of W:", jnp.linalg.eigvals(W))
        b = 2e-1 * jnp.array([-0.5, 0.5])
        y0 = jnp.array([1.0, 0.5, 0.0, 0.0])
    case _:
        raise NotImplementedError


def apply_eps_to_diagonal(A: jax.Array, eps: float = 1e-6) -> jax.Array:
    """
    Add a small number to the diagonal to avoid singularities
    """
    # extract the diagonal
    diag_A = jnp.diag(A)

    # get the sign of the diagonal
    diag_A_sign = jnp.sign(diag_A)
    # set zero sign to 1 (i.e. positive)
    diag_A_sign = jnp.where(diag_A_sign == 0, 1, diag_A_sign)
    # add eps to the diagonal
    diag_A_epsed = lax.select(
        jnp.abs(diag_A) < eps,
        diag_A_sign * eps,
        diag_A,
    )

    # update the matrix
    A_epsed = A + (diag_A_epsed - jnp.diag(A))

    return A_epsed


K, D = apply_eps_to_diagonal(K), apply_eps_to_diagonal(D)


if jnp.all(jnp.diag(D) == 0.0):
    print("Undamped oscillators")
elif jnp.all(jnp.diag(D) < 2 * jnp.sqrt(m * jnp.diag(K))):
    print("Underdamped oscillators")
elif jnp.all(jnp.diag(D) == 2 * jnp.sqrt(m * jnp.diag(K))):
    print("Critically damped oscillators")
elif jnp.all(jnp.diag(D) > 2 * jnp.sqrt(m * jnp.diag(K))):
    print("Overdamped oscillators")
else:
    raise ValueError


def ode_fn(
    t: jax.Array,
    y: jax.Array,
    *args,
    m: jax.Array,
    K: jax.Array,
    D: jax.Array,
    W: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """
    Harmonic oscillator ODE.
    Args:
        t: time
        y: oscillator state
        m: mass
        K: stiffness matrix
        D: damping matrix
        W: coupling matrix
        b: bias
    Returns:
        y_d: derivative of the oscillator state
    """
    x, x_d = jnp.split(y, 2)
    x_dd = m ** (-1) * (-K @ x - D @ x_d - jnp.tanh(W @ x + b))
    y_d = jnp.concatenate([x_d, x_dd])
    return y_d


def closed_form_approximation_step_no_damping(
    t: jax.Array,
    t0: jax.Array,
    y0: jax.Array,
    m: jax.Array,
    k: jax.Array,
    f_ext: jax.Array,
) -> jax.Array:
    """
    Closed-form solution of the harmonic oscillator with underdamping.
    https://scholar.harvard.edu/files/schwartz/files/lecture1-oscillators-and-linearity.pdf
    Args:
        t: time
        t0: initial time
        y0: initial state
        m: mass
        k: stiffness vector
        f_ext: external force
    Returns:
        y: oscillator state at time t
    """
    x0, v0 = jnp.split(y0, 2)

    # natural frequency
    omega_n = jnp.sqrt(k / m)

    # constants for the closed-form solution
    ctilde1 = x0 - f_ext / k
    ctilde2 = v0 / omega_n

    x = (
        ctilde1 * jnp.cos(omega_n * (t - t0)) + ctilde2 * jnp.sin(omega_n * (t - t0))
    ) + f_ext / k
    x_d = -(
        (-ctilde2 * omega_n) * jnp.cos(omega_n * (t - t0))
        + (ctilde1 * omega_n) * jnp.sin(omega_n * (t - t0))
    )

    y = jnp.concatenate([x, x_d]).astype(jnp.float64)
    return y


def closed_form_approximation_step(
    t: jax.Array,
    t0: jax.Array,
    y0: jax.Array,
    m: jax.Array,
    k: jax.Array,
    d: jax.Array,
    f_ext: jax.Array,
) -> jax.Array:
    """
    Closed-form solution of the harmonic oscillator.
    https://scholar.harvard.edu/files/schwartz/files/lecture1-oscillators-and-linearity.pdf
    Args:
        t: time
        t0: initial time
        y0: initial state
        m: mass
        k: stiffness vector
        d: damping vector
        f_ext: external force
    Returns:
        y: oscillator state at time t
    """
    x0, v0 = jnp.split(y0, 2)

    # cast to complex numbers
    x0, v0 = x0.astype(jnp.complex128), v0.astype(jnp.complex128)
    m, k, d = m.astype(jnp.complex128), k.astype(jnp.complex128), d.astype(jnp.complex128)
    f_ext = f_ext.astype(jnp.complex128)

    # natural frequency
    omega_n = jnp.sqrt(k / m)
    # damping ratio
    zeta = d / (2 * jnp.sqrt(m * k))

    # https://tttapa.github.io/Pages/Arduino/Audio-and-Signal-Processing/VU-Meters/Damped-Harmonic-Oscillator.html
    alpha = zeta * omega_n
    beta = omega_n * jnp.sqrt(1 - zeta**2)
    lambda1, lambda2 = -alpha + beta * 1j, -alpha - beta * 1j

    # when d = 2 * sqrt(m * k) => zeta = 1 => lambda1 = lambda2 => lambda2-lambda1 = 0, the system is critically damped
    # theoretically, the solution would be different. However, this case will rarely happen in practice
    # therefore, we will just try to prevent the division by zero
    lambda_diff = lambda2 - lambda1
    lambda_diff_epsed = apply_eps_to_array(lambda_diff)

    # constants for the closed-form solution
    """
    c1 = (-v0 + lambda2 * (x0 - f_ext / k)) / lambda_diff_epsed
    c2 = (v0 - lambda1 * (x0 - f_ext / k)) / lambda_diff_epsed
    ctilde1 = c1 + c2
    ctilde2 = (c1 - c2) * 1j
    """
    ctilde1 = x0 - f_ext / k
    ctilde2 = (
        (-2 * v0 + (lambda1 + lambda2) * (x0 - f_ext / k)) / lambda_diff_epsed * 1j
    )

    x = (
        ctilde1 * jnp.cos(beta * (t - t0)) + ctilde2 * jnp.sin(beta * (t - t0))
    ) * jnp.exp(-(alpha * (t - t0))) + f_ext / k
    x_d = -(
        (ctilde1 * alpha - ctilde2 * beta) * jnp.cos(beta * (t - t0))
        + (ctilde1 * beta + ctilde2 * alpha) * jnp.sin(beta * (t - t0))
    ) * jnp.exp(-alpha * (t - t0))

    y = jnp.real(jnp.concatenate([x, x_d])).astype(jnp.float64)
    return y


def simulate_closed_form_approximation(
    ts: jax.Array,
    y0: jax.Array,
    readout_dt: jax.Array,
    m: jax.Array,
    K: jax.Array,
    D: jax.Array,
    W: jax.Array,
    b: jax.Array,
):
    # assume constant time step
    sim_dt = ts[1] - ts[0]
    ts_dt_template = jnp.arange(0.0, sim_dt + readout_dt, readout_dt)

    if jnp.all(jnp.diag(D) == 0.0):
        closed_form_approximation_step_fn = partial(
            closed_form_approximation_step_no_damping, m=m, k=jnp.diag(K)
        )
    else:
        closed_form_approximation_step_fn = partial(
            closed_form_approximation_step, m=m, k=jnp.diag(K), d=jnp.diag(D)
        )

    def approx_step_fn(
        carry: Dict[str, jax.Array], input: Dict[str, jax.Array]
    ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
        y = carry["y"]
        x, x_d = jnp.split(y, 2)

        f_ext = (
            -(K - jnp.diag(jnp.diag(K))) @ x
            - (D - jnp.diag(jnp.diag(D))) @ x_d
            - jnp.tanh(W @ x + b)
        )

        ts_dt = ts_dt_template + carry["t"]
        f_ext_ts = jnp.repeat(f_ext[None, :], ts_dt.shape[0], axis=0)
        y_ts_dt = jax.vmap(
            partial(closed_form_approximation_step_fn, t0=ts_dt[0], y0=y, f_ext=f_ext)
        )(ts_dt)

        carry = dict(t=ts_dt[-1], y=y_ts_dt[-1])
        step_data = dict(ts=ts_dt[:-1], y_ts=y_ts_dt[:-1], f_ext_ts=f_ext_ts[:-1])
        return carry, step_data

    input_ts = dict(
        ts=ts,
    )

    carry = dict(
        t=ts[0],
        y=y0,
    )

    carry, sim_ts = lax.scan(approx_step_fn, carry, input_ts)

    return sim_ts


# Define the harmonic oscillator ODE term
ode_term = ODETerm(
    partial(ode_fn, m=m, K=K, D=D, W=W, b=b),
)

sol_numerical_high_precision = diffeqsolve(
    ode_term,
    Euler(),
    t0=ts[0],
    t1=ts[-1],
    dt0=dt,
    y0=y0,
    saveat=SaveAt(ts=ts),
    max_steps=ts.shape[-1],
)
y_ts_numerical_high_precision = sol_numerical_high_precision.ys

low_precision_dt = 1e3 * dt
sol_numerical_low_precision = diffeqsolve(
    ode_term,
    Euler(),
    t0=ts[0],
    t1=ts[-1],
    dt0=low_precision_dt,
    y0=y0,
    saveat=SaveAt(ts=ts),
    max_steps=ts.shape[-1],
)
y_ts_numerical_low_precision = sol_numerical_low_precision.ys

# evaluate the closed-form solution
closed_form_dt = 1e3 * dt
ts_sim_closed_form = jnp.arange(ts[0], ts[-1], 1e3 * dt)
closed_form_sim_ts = simulate_closed_form_approximation(
    ts_sim_closed_form, y0, readout_dt=dt, m=m, K=K, D=D, W=W, b=b
)
for key in closed_form_sim_ts.keys():
    closed_form_sim_ts[key] = closed_form_sim_ts[key].reshape(
        (-1,) + closed_form_sim_ts[key].shape[2:]
    )
ts_closed_form = closed_form_sim_ts["ts"]
y_ts_closed_form = closed_form_sim_ts["y_ts"]


# plot the position
plt.plot(
    ts,
    y_ts_numerical_high_precision[:, :num_units],
    label=rf"Numerical solution with dt = {dt}s",
    linestyle="--",
    linewidth=2.5,
)
plt.gca().set_prop_cycle(None)
plt.plot(
    ts,
    y_ts_numerical_low_precision[:, :num_units],
    label=rf"Numerical solution with dt = {low_precision_dt}s",
    linestyle=":",
    linewidth=2.0,
)
plt.gca().set_prop_cycle(None)
plt.plot(
    ts_closed_form,
    y_ts_closed_form[:, :num_units:],
    label=rf"Closed-form solution dt = {closed_form_dt}s",
)
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.grid()
plt.box(True)
plt.title("Coupled oscillator position")
plt.show()

# plot the velocity
plt.plot(
    ts,
    y_ts_numerical_high_precision[:, num_units:],
    label=rf"Numerical solution with dt = {dt}s",
    linestyle="--",
    linewidth=2.5,
)
plt.gca().set_prop_cycle(None)
plt.plot(
    ts,
    y_ts_numerical_low_precision[:, num_units:],
    label=rf"Numerical solution with dt = {low_precision_dt}s",
    linestyle=":",
    linewidth=2.0,
)
plt.gca().set_prop_cycle(None)
plt.plot(
    ts_closed_form,
    y_ts_closed_form[:, num_units:],
    label=rf"Closed-form solution dt = {closed_form_dt}s",
)
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()
plt.grid()
plt.box(True)
plt.title("Coupled oscillator velocity")
plt.show()

import jax

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_platform_name", "cpu")  # use CPU
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from functools import partial
from jax import jit, lax, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import timeit
from typing import Callable, Dict, Optional, Tuple

from src.dynamics.utils import apply_eps_to_array


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)
figsize = (5.0, 3.0)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# time steps
dt_readout = jnp.array(1e-2)
dt_high_precision = jnp.array(5e-5)
dt_low_precision_tsit = jnp.array(1e-1)
dt_low_precision_euler = jnp.array(5e-2)
ts_readout = jnp.arange(0.0, 60.0, dt_readout)
dt_closed_form = 1e-1

# output directory
outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

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


# K, D = apply_eps_to_diagonal(K), apply_eps_to_diagonal(D)
def classify_damping_regime(m: jax.Array, K: jax.Array, D: jax.Array) -> str:
    if jnp.all(jnp.diag(D) == 0.0):
        print("Undamped oscillators")
        return "undamped"
    elif jnp.all(jnp.diag(D) < 2 * jnp.sqrt(m * jnp.diag(K))):
        print("Underdamped oscillators")
        return "underdamped"
    elif jnp.all(jnp.diag(D) == 2 * jnp.sqrt(m * jnp.diag(K))):
        print("Critically damped oscillators")
        return "critically-damped"
    elif jnp.all(jnp.diag(D) > 2 * jnp.sqrt(m * jnp.diag(K))):
        print("Overdamped oscillators")
        return "overdamped"
    else:
        print("General damping regime")
        return "general"


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
    **kwargs
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
    # x0, v0 = x0.astype(jnp.complex128), v0.astype(jnp.complex128)
    m, k, d = m.astype(jnp.complex128), k.astype(jnp.complex128), d.astype(jnp.complex128)
    # f_ext = f_ext.astype(jnp.complex128)

    # natural frequency
    omega_n = jnp.sqrt(k / m)
    # damping ratio
    zeta = d / (2 * jnp.sqrt(m * k))

    # https://tttapa.github.io/Pages/Arduino/Audio-and-Signal-Processing/VU-Meters/Damped-Harmonic-Oscillator.html
    alpha = zeta * omega_n
    beta = omega_n * jnp.sqrt(1 - zeta**2)
    # lambda1, lambda2 = -alpha + beta * 1j, -alpha - beta * 1j

    # when d = 2 * sqrt(m * k) => zeta = 1 => lambda1 = lambda2 => lambda2-lambda1 = 0, the system is critically damped
    # theoretically, the solution would be different. However, this case will rarely happen in practice
    # therefore, we will just try to prevent the division by zero
    # lambda_diff = lambda2 - lambda1
    lambda_diff = -2 * beta * 1j
    lambda_diff_epsed = apply_eps_to_array(lambda_diff)

    # constants for the closed-form solution
    """
    ctilde1 = x0 - f_ext / k
    ctilde2 = (
        (-2 * v0 + (lambda1 + lambda2) * (x0 - f_ext / k)) / lambda_diff_epsed * 1j
    )
    """
    ctilde1 = x0 - f_ext / k
    ctilde2 = -2j / lambda_diff_epsed * (v0 + alpha * (x0 - f_ext / k))

    # time constant
    _dt = t - t0

    x = (
        ctilde1 * jnp.cos(beta * _dt) + ctilde2 * jnp.sin(beta * _dt)
    ) * jnp.exp(-(alpha * _dt)) + f_ext / k
    x_d = -(
        (ctilde1 * alpha - ctilde2 * beta) * jnp.cos(beta * _dt)
        + (ctilde1 * beta + ctilde2 * alpha) * jnp.sin(beta * _dt)
    ) * jnp.exp(-alpha * (t - t0))

    y = jnp.real(jnp.concatenate([x, x_d])).astype(jnp.float64)
    return y


def closed_form_approximation_step_underdamped(
    t: jax.Array,
    t0: jax.Array,
    y0: jax.Array,
    m: jax.Array,
    k: jax.Array,
    d: jax.Array,
    f_ext: jax.Array,
) -> jax.Array:
    """
    Closed-form solution of the harmonic oscillator specialized for the underdamped case.
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
    # time constant
    _dt = t - t0

    # cast to half precision
    x0, v0 = x0.astype(jnp.float32), v0.astype(jnp.float32)
    m, k, d = m.astype(jnp.float32), k.astype(jnp.float32), d.astype(jnp.float32)
    _dt, f_ext = _dt.astype(jnp.float32), f_ext.astype(jnp.float32)

    # natural frequency
    omega_n = jnp.sqrt(k / m)
    # damping ratio
    zeta = d / (2 * jnp.sqrt(m * k))

    # https://tttapa.github.io/Pages/Arduino/Audio-and-Signal-Processing/VU-Meters/Damped-Harmonic-Oscillator.html
    alpha = zeta * omega_n
    beta = omega_n * jnp.sqrt(1 - zeta**2)

    # constants for the closed-form solution
    ctilde1 = x0 - f_ext / k
    ctilde2 = (v0 + alpha * (x0 - f_ext / k)) / beta

    alpha_exp = jnp.exp(-alpha * _dt)
    beta_dt_prod = beta * _dt
    beta_cos = jnp.cos(beta_dt_prod)
    beta_sin = jnp.sin(beta_dt_prod)

    x = (ctilde1 * beta_cos + ctilde2 * beta_sin) * alpha_exp + f_ext / k
    x_d = -(
        (ctilde1 * alpha - ctilde2 * beta) * beta_cos
        + (ctilde1 * beta + ctilde2 * alpha) * beta_sin
    ) * alpha_exp

    y = jnp.concatenate([x, x_d]).astype(jnp.float64)
    return y


def cfa_factory(
    ts_sim: jax.Array,
    readout_dt: Optional[jax.Array] = None,
    damping_regime: Optional[str] = "general"
) -> Callable:
    """
    Arguments:
        ts_sim: time steps for the simulation
        readout_dt: time steps for the readout. If None, the readout is not performed and only the last state is returned.
        damping_regime: the damping regime of the system. Options: "general", "underdamped", None. If None, the system is assumed to be undamped.
    """
    def simulate_closed_form_approximation(
        _y0: jax.Array,
        _m: jax.Array,
        _K: jax.Array,
        _D: jax.Array,
        _W: jax.Array,
        _b: jax.Array,
    ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
        cfa_step_fn = partial(
            closed_form_approximation_step_fn , m=_m, k=jnp.diag(_K), d=jnp.diag(_D)
        )

        def approx_step_fn(
                carry: Dict[str, jax.Array], input: Dict[str, jax.Array]
        ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
            t, y = carry["t"], carry["y"]
            x, x_d = jnp.split(y, 2)

            f_ext = (
                    -(_K - jnp.diag(jnp.diag(_K))) @ x
                    - (_D - jnp.diag(jnp.diag(_D))) @ x_d
                    - jnp.tanh(_W @ x + _b)
            )

            ts_readout_dt = ts_readout_dt_template + carry["t"]
            f_ext_ts = jnp.repeat(f_ext[None, :], ts_readout_dt.shape[0], axis=0)
            y_ts_dt = jax.vmap(
                partial(cfa_step_fn, t0=ts_readout_dt[0], y0=y, f_ext=f_ext)
            )(ts_readout_dt)
            carry = dict(t=ts_readout_dt[-1], y=y_ts_dt[-1])
            
            step_data = {}
            if readout_dt is not None:
                step_data = dict(ts=ts_readout_dt[:-1], y_ts=y_ts_dt[:-1], f_ext_ts=f_ext_ts[:-1])

            """
            y_next = cfa_step_fn(t0=t, y0=y, t=input["ts"], f_ext=f_ext)
            carry = dict(t=input["ts"], y=y_next)

            step_data = {}
            if readout_dt is not None:
                step_data = dict(ts=input["ts"], y_ts=y_next, f_ext_ts=f_ext)
            """

            return carry, step_data

        input_ts = dict(
            ts=ts_sim,
        )

        carry = dict(
            t=ts_sim[0],
            y=_y0,
        )

        carry, sim_ts = lax.scan(approx_step_fn, carry, input_ts)

        return carry, sim_ts

    match damping_regime:
        case "general":
            closed_form_approximation_step_fn = closed_form_approximation_step
        case "underdamped":
            closed_form_approximation_step_fn = closed_form_approximation_step_underdamped
        case None:
            closed_form_approximation_step_fn = closed_form_approximation_step_no_damping
        case _:
            raise ValueError

    # assume constant time step
    sim_dt = ts_sim[1]-ts_sim[0]
    if readout_dt is None:
        ts_readout_dt_template = jnp.arange(0.0, 2 * sim_dt, sim_dt)
    else:
        ts_readout_dt_template = jnp.arange(0.0, sim_dt + readout_dt, readout_dt)

    return simulate_closed_form_approximation


def euler_factory(
    ts_sim: jax.Array,
    readout: bool = True,
) -> Callable:
    def simulate(
        _y0: jax.Array,
        _m: jax.Array,
        _K: jax.Array,
        _D: jax.Array,
        _W: jax.Array,
        _b: jax.Array,
    ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
        ode_bound_fn = partial(
            ode_fn, m=_m, K=_K, D=_D, W=_W, b=_b
        )

        def approx_step_fn(
                carry: Dict[str, jax.Array], input: Dict[str, jax.Array]
        ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
            y = carry["y"]
            t_next = input["ts"]

            y_d = ode_bound_fn(t_next, y)
            # integrate with Euler
            y_next = y + (t_next - carry["t"]) * y_d

            carry = dict(t=t_next, y=y_next)

            step_data = {}
            if readout is True:
                step_data = dict(ts=t_next, y_ts=y_next)

            return carry, step_data

        input_ts = dict(
            ts=ts_sim,
        )

        carry = dict(
            t=ts_sim[0],
            y=_y0,
        )

        carry, sim_ts = lax.scan(approx_step_fn, carry, input_ts)

        return carry, sim_ts

    return simulate


def plot_single_rollout():
    # parameters
    num_units = 3
    ts_readout = jnp.arange(0.0, 40.0, dt_readout)
    m = 1.0 * jnp.ones((num_units,))  # mass
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
        case 3:
            """
            K = 0.1 * jnp.array([[1.0, 0.25, 0.15], [0.25, 1.0, -0.1], [0.15, -0.1, 1.0]])
            D = 0.05 * jnp.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
            W = 2e-1 * jnp.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
            print("Eigvals of W:", jnp.linalg.eigvals(W))
            b = 2e-1 * jnp.array([-0.5, 0.5, 0.0])
            y0 = jnp.array([1.0, 0.7, 0.4, 0.0, 0.0, 0.0])
            """
            omega_n = jnp.array([0.4, 0.3, 0.5])
            K = jnp.diag(jnp.array([1.5, 1.8, 2.0]))
            m = jnp.diag(K) / omega_n ** 2
            zeta = jnp.array([0.1, 0.05, 0.02])
            D = jnp.diag(2 * zeta * jnp.sqrt(jnp.diag(K) * m))
            W = jnp.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
            b = jnp.array([-0.5, 0.5, 0.0])
            y0 = jnp.array([1.0, 0.7, 0.4, 0.0, 0.0, 0.0])
        case 4:
            m = 0.02 * jnp.ones((num_units,))
            K = 0.5 * jnp.eye(num_units)
            D = jnp.diag(jnp.sqrt(jnp.diag(K) * m))
            W = 1e0 * jnp.eye(num_units)
            b = jnp.zeros((num_units,))
            y0 = jnp.array([1.0, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
        case _:
            raise ValueError("Unsupported number of units")

    damping_regime = classify_damping_regime(m, K, D)
    omega_n = jnp.sqrt(jnp.diag(K) / m)
    print("Natural frequencies:", omega_n)

    # Define the harmonic oscillator ODE term
    ode_term = ODETerm(
        partial(ode_fn, m=m, K=K, D=D, W=W, b=b),
    )
    simulation_fn_high_precision = jit(partial(
        diffeqsolve,
        ode_term,
        Tsit5(),
        t0=ts_readout[0],
        t1=ts_readout[-1],
        dt0=dt_high_precision,
        y0=y0,
        saveat=SaveAt(ts=ts_readout),
        max_steps=None,
    ))
    sol_numerical_high_precision = simulation_fn_high_precision()
    y_ts_numerical_high_precision = sol_numerical_high_precision.ys

    simulation_fn_low_precision_tsit = jit(partial(
        diffeqsolve,
        ode_term,
        Tsit5(),
        t0=ts_readout[0],
        t1=ts_readout[-1],
        dt0=dt_low_precision_tsit,
        y0=y0,
        saveat=SaveAt(ts=ts_readout),
        max_steps=None,
    ))
    sol_numerical_low_precision_tsit = simulation_fn_low_precision_tsit()
    y_ts_numerical_low_precision_tsit = sol_numerical_low_precision_tsit.ys

    """
    simulation_fn_low_precision_euler = jit(partial(
        diffeqsolve,
        ode_term,
        Euler(),
        t0=ts_readout[0],
        t1=ts_readout[-1],
        dt0=dt_low_precision,
        y0=y0,
        saveat=SaveAt(ts=ts_readout),
        max_steps=None,
    ))
    sol_numerical_low_precision_euler = simulation_fn_low_precision_euler()
    ts_low_precision_euler = ts_readout
    y_ts_numerical_low_precision_euler = sol_numerical_low_precision_euler.ys
    """
    ts_sim_low_precision_euler = jnp.arange(ts_readout[0], ts_readout[-1], dt_low_precision_euler)
    simulation_fn_low_precision_euler = euler_factory(ts_sim_low_precision_euler, readout=True)
    _, sim_ts_low_precision_euler = simulation_fn_low_precision_euler(y0, _m=m, _K=K, _D=D, _W=W, _b=b)
    ts_low_precision_euler  = sim_ts_low_precision_euler["ts"]
    y_ts_numerical_low_precision_euler = sim_ts_low_precision_euler["y_ts"]


    # evaluate the closed-form solution
    ts_sim_closed_form = jnp.arange(ts_readout[0], ts_readout[-1], dt_closed_form)
    simulate_closed_form_approximation_fn = jit(partial(
        cfa_factory(ts_sim_closed_form, readout_dt=dt_readout), y0, m, K, D, W, b
    ))
    _, closed_form_sim_ts = simulate_closed_form_approximation_fn()
    for key in closed_form_sim_ts.keys():
        closed_form_sim_ts[key] = closed_form_sim_ts[key].reshape(
            (-1,) + closed_form_sim_ts[key].shape[2:]
        )
    ts_closed_form = closed_form_sim_ts["ts"]
    y_ts_closed_form = closed_form_sim_ts["y_ts"]

    linewidth_solid = 1.8
    linewidth_dotted = 2.5
    # plot the position
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(num_units):
        """
        markevery = 50
        markersize = 5
        marker="o",
        markevery=markevery,
        markersize=markersize,
        """
        line_high_precision, = ax.plot(
            ts_readout,
            y_ts_numerical_high_precision[:, i],
            linestyle=":",
            linewidth=linewidth_dotted,
            color="black",
            label=rf"CON Tsit5 $\delta t = {dt_high_precision}$s",
        )
        """
        line_low_precision_tsit, = ax.plot(
            ts_readout,
            y_ts_numerical_low_precision_tsit[:, i],
            linewidth=linewidth_solid,
            color=colors[0],
            label=rf"CON Tsit5 $\delta t = {dt_low_precision_tsit}$s",
        )
        """
        line_low_precision_euler, = ax.plot(
            ts_low_precision_euler,
            y_ts_numerical_low_precision_euler[:, i],
            linewidth=linewidth_solid,
            color=colors[1],
            label=rf"CON Euler $\delta t = {dt_low_precision_euler}$s",
        )
        line_cfa, = ax.plot(
            ts_closed_form,
            y_ts_closed_form[:, i],
            linewidth=linewidth_solid,
            color=colors[2],
            label=rf"CFA-CON $\delta t = {dt_closed_form}$s",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend(handles=[line_high_precision, line_low_precision_euler, line_cfa], loc="lower left")
    plt.grid(True)
    plt.box(True)
    # plt.title("Coupled oscillator position")
    plt.tight_layout()
    plt.savefig(outputs_dir / "coupled_oscillator_position.pdf")
    plt.show()

    # plot the velocity
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(num_units):
        line_high_precision, = ax.plot(
            ts_readout,
            y_ts_numerical_high_precision[:, num_units + i],
            linestyle=":",
            linewidth=linewidth_dotted,
            color="black",
            label=rf"CON Tsit5 $\delta t = {dt_high_precision}$s",
        )
        """
        line_low_precision_tsit, = ax.plot(
            ts_readout,
            y_ts_numerical_low_precision_tsit[:, num_units + i],
            linewidth=linewidth_solid,
            color=colors[0],
            label=rf"CON Tsit5 $\delta t = {dt_low_precision_tsit}$s",
        )
        """
        line_low_precision_euler, = ax.plot(
            ts_low_precision_euler,
            y_ts_numerical_low_precision_euler[:, num_units + i],
            linewidth=linewidth_solid,
            color=colors[1],
            label=rf"CON Euler $\delta t = {dt_low_precision_euler}$s",
        )
        line_cfa, = ax.plot(
            ts_closed_form,
            y_ts_closed_form[:, num_units + i],
            linewidth=linewidth_solid,
            color=colors[2],
            label=rf"CFA-CON $\delta t = {dt_closed_form}$s",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend(handles=[line_high_precision, line_low_precision_euler, line_cfa], loc="lower left")
    plt.grid(True)
    plt.box(True)
    # plt.title("Coupled oscillator position")
    plt.tight_layout()
    plt.savefig(outputs_dir / "coupled_oscillator_velocity.pdf")
    plt.show()


def benchmark_sim_to_real_time_factor():
    # parameters
    num_units = 50
    m = 1.0 * jnp.ones((num_units,))  # mass
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
        case 3:
            K = 0.1 * jnp.array([[1.0, 0.25, 0.15], [0.25, 1.0, -0.1], [0.15, -0.1, 1.0]])
            D = 0.05 * jnp.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
            W = 2e-1 * jnp.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
            print("Eigvals of W:", jnp.linalg.eigvals(W))
            b = 2e-1 * jnp.array([-0.5, 0.5, 0.0])
            y0 = jnp.array([1.0, 0.7, 0.4, 0.0, 0.0, 0.0])
        case _:
            K = 0.1 * jnp.eye(num_units)
            D = 0.05 * jnp.eye(num_units)
            W = 2e-1 * jnp.eye(num_units)
            b = jnp.zeros((num_units,))
            y0 = jnp.ones((2 * num_units,))

    classify_damping_regime(m, K, D)

    # Define the harmonic oscillator ODE term
    ode_term = ODETerm(
        partial(ode_fn, m=m, K=K, D=D, W=W, b=b),
    )
    simulation_high_precision_fn = jit(partial(
        diffeqsolve,
        ode_term,
        Tsit5(),
        t0=ts_readout[0],
        t1=ts_readout[-1],
        dt0=dt_high_precision,
        y0=y0,
        # saveat=SaveAt(ts=ts_readout),
        max_steps=None,
    ))

    simulation_low_precision_tsit_fn = jit(partial(
        diffeqsolve,
        ode_term,
        Tsit5(),
        t0=ts_readout[0],
        t1=ts_readout[-1],
        dt0=dt_low_precision_tsit,
        y0=y0,
        # saveat=SaveAt(ts=ts_readout),
        max_steps=None,
    ))

    """
    simulation_low_precision_euler_fn = jit(partial(
        diffeqsolve,
        ode_term,
        Euler(),
        t0=ts_readout[0],
        t1=ts_readout[-1],
        dt0=dt_low_precision,
        y0=y0,
        # saveat=SaveAt(ts=ts_readout),
        max_steps=None,
    ))
    """
    ts_sim_low_precision_euler = jnp.arange(ts_readout[0], ts_readout[-1], dt_low_precision_euler)
    simulation_low_precision_euler_fn = jit(partial(
        euler_factory(ts_sim_low_precision_euler, readout=False),
        y0, _m=m, _K=K, _D=D, _W=W, _b=b
    ))
    simulation_low_precision_euler_fn()

    # evaluate the closed-form solution
    ts_sim_closed_form = jnp.arange(ts_readout[0], ts_readout[-1], dt_closed_form)
    simulate_closed_form_approximation_fn = jit(partial(
        cfa_factory(ts_sim_closed_form, readout_dt=None),y0, m, K, D, W, b
    ))

    # evaluate the closed-form solution in the underdamped regime
    ts_sim_closed_form = jnp.arange(ts_readout[0], ts_readout[-1], dt_closed_form)
    simulate_closed_form_approximation_underdamped_fn = jit(partial(
        cfa_factory(ts_sim_closed_form, readout_dt=None, damping_regime="underdamped"), y0, m, K, D, W, b
    ))

    # make sure all functions are compiled
    # simulation_high_precision_fn()
    simulation_low_precision_tsit_fn()
    simulation_low_precision_euler_fn()
    simulate_closed_form_approximation_fn()
    simulate_closed_form_approximation_underdamped_fn()

    # benchmark the computational time
    number_of_repeats = 10
    number_of_runs = 100
    # repeats a list of the time for each repeat
    time_high_precision = timeit.timeit(
        simulation_high_precision_fn,
        number=5,
    )
    time_low_precision_tsit_ls = jnp.array(timeit.repeat(
        simulation_low_precision_tsit_fn,
        repeat=number_of_repeats,
        number=number_of_runs,
    ))
    time_low_precision_euler_ls = jnp.array(timeit.repeat(
        simulation_low_precision_euler_fn,
        repeat=number_of_repeats,
        number=number_of_runs,
    ))
    time_cfa_ls = jnp.array(timeit.repeat(
        simulate_closed_form_approximation_fn,
        repeat=number_of_repeats,
        number=number_of_runs,
    ))
    time_cfa_underdamped_ls = jnp.array(timeit.repeat(
        simulate_closed_form_approximation_underdamped_fn,
        repeat=number_of_repeats,
        number=number_of_runs,
    ))
    # take the minimum computation time as recommended here: https://docs.python.org/3/library/timeit.html
    s2rr_high_precision = 5 * (ts_readout[-1] - ts_readout[0]) / time_high_precision
    s2rr_low_precision_tsit = number_of_runs * (ts_readout[-1] - ts_readout[0]) / jnp.min(time_low_precision_tsit_ls)
    s2rr_low_precision_euler = number_of_runs * (ts_readout[-1] - ts_readout[0]) / jnp.min(time_low_precision_euler_ls)
    s2rr_cfa = number_of_runs * (ts_readout[-1] - ts_readout[0]) / jnp.min(time_cfa_ls)
    s2rr_cfa_underdamped = number_of_runs * (ts_readout[-1] - ts_readout[0]) / jnp.min(time_cfa_underdamped_ls)
    print(
        f"High precision: {time_high_precision:.4f}s, simulation to real-time factor: {s2rr_high_precision:.2f}x"
    )
    print(
        f"Low precision Tsit5: {jnp.min(time_low_precision_tsit_ls):.4f}s, "
        f"simulation to real-time factor: {s2rr_low_precision_tsit:.0f}x"
    )
    print(
        f"Low precision Euler: {jnp.min(time_low_precision_euler_ls):.4f}s, "
        f"simulation to real-time factor: {s2rr_low_precision_euler:.0f}x"
    )
    print(
        f"Closed-form approximation: {jnp.min(time_cfa_ls):.4f}s, real-time ratio: {s2rr_cfa:.0f}x"
    )
    print(
        f"Closed-form approximation underdamped: {jnp.min(time_cfa_underdamped_ls):.4f}s, real-time ratio: {s2rr_cfa_underdamped:.0f}x"
    )


def benchmark_integration_error():
    num_samples = 100
    num_units = 50
    enforce_underdamping = False
    plot_solution = False

    # set random seed
    rng = jax.random.PRNGKey(0)
    sample_idx = -1

    benchmarking_data = dict(
        low_precision_tsit_mse=[],
        low_precision_euler_mse=[],
        cfa_mse=[],
        cfa_underdamped_mse=[]
    )
    while sample_idx < (num_samples-1):
        sample_idx += 1
        print(f"Sample {sample_idx + 1}/{num_samples}")

        # split the rng key
        rng, key1, key2, key3, key4, key5, key6 = random.split(rng, num=7)

        # sample the parameters
        """
        L_K = jnp.tril(random.uniform(key1, shape=(num_units, num_units), minval=-1.0, maxval=1.0))  # lower triangular matrix
        K = L_K @ L_K.T  # positive definite matrix
        L_D = jnp.tril(random.uniform(key1, shape=(num_units, num_units), minval=-1.0, maxval=1.0))  # lower triangular matrix
        # make sure that the diagional elements are positive
        L_D = L_D.at[jnp.diag_indices(num_units)].set(jnp.abs(jnp.diag(L_D)))
        # make sure that the diagional elements are positive
        L_D = L_D.at[jnp.diag_indices(num_units)].set(jnp.abs(jnp.diag(L_D)))
        D = L_D @ L_D.T  # positive definite matrix
        """
        # sample the natural frequencies
        omega_n = random.uniform(key1, shape=(num_units,), minval=5e-2, maxval=5e-1)
        # print("Natural frequencies:\n", omega_n)
        # sample the stiffnesses
        K = jnp.diag(random.uniform(key1, shape=(num_units,), minval=2e-1, maxval=2e0))
        # print("Stiffnesses:\n", jnp.diag(K))
        # sample the masses
        m = jnp.diag(K) / omega_n ** 2
        # print("Masses:\n", m)
        # sample the damping coefficients
        # K = jnp.diag(random.uniform(key2, shape=(num_units,), minval=2e-2, maxval=2e-1))
        if enforce_underdamping is True:
            zeta = random.uniform(key3, shape=(num_units,), minval=0.1, maxval=0.9)
            D = jnp.diag(2 * zeta * jnp.sqrt(m * jnp.diag(K)))
        else:
            zeta = random.uniform(key3, shape=(num_units,), minval=0.1, maxval=2.0)
            D = jnp.diag(2 * zeta * jnp.sqrt(m * jnp.diag(K)))
        # print("Damping ratios:\n", zeta)

        L_W = random.uniform(key4, shape=(num_units, num_units), minval=-1.0, maxval=1.0)  # lower triangular matrix
        # make sure that the diagional elements are positive
        L_W = L_W.at[jnp.diag_indices(num_units)].set(jnp.abs(jnp.diag(L_W)))
        W = L_W @ L_W.T  # hyperbolic coupling matrix
        b = random.uniform(key5, shape=(num_units,), minval=-1.0, maxval=1.0)
        # sample the initial state
        y0 = random.uniform(key6, shape=(2 * num_units,), minval=-1.0, maxval=1.0)

        # check the damping regime
        damping_regime = classify_damping_regime(m, K, D)

        # Define the harmonic oscillator ODE term
        ode_term = ODETerm(
            partial(ode_fn, m=m, K=K, D=D, W=W, b=b),
        )
        sol_high_precision = diffeqsolve(
            ode_term,
            Tsit5(),
            t0=ts_readout[0],
            t1=ts_readout[-1],
            dt0=dt_high_precision,
            y0=y0,
            saveat=SaveAt(ts=ts_readout),
            max_steps=None,
        )
        y_ts_numerical_high_precision = sol_high_precision.ys

        sol_low_precision_tsit = diffeqsolve(
            ode_term,
            Tsit5(),
            t0=ts_readout[0],
            t1=ts_readout[-1],
            dt0=dt_low_precision_tsit,
            y0=y0,
            saveat=SaveAt(ts=ts_readout),
            max_steps=None,
        )
        y_ts_numerical_low_precision_tsit = sol_low_precision_tsit.ys

        sol_low_precision_euler = diffeqsolve(
            ode_term,
            Euler(),
            t0=ts_readout[0],
            t1=ts_readout[-1],
            dt0=dt_low_precision_euler,
            y0=y0,
            saveat=SaveAt(ts=ts_readout),
            max_steps=None,
        )
        y_ts_numerical_low_precision_euler = sol_low_precision_euler.ys

        # evaluate the closed-form solution
        ts_sim_closed_form = jnp.arange(ts_readout[0], ts_readout[-1], dt_closed_form)
        _, closed_form_sim_ts = cfa_factory(
            ts_sim_closed_form, readout_dt=jnp.array(dt_readout)
        )(y0, m, K, D, W, b)
        for key in closed_form_sim_ts.keys():
            closed_form_sim_ts[key] = closed_form_sim_ts[key].reshape(
                (-1,) + closed_form_sim_ts[key].shape[2:]
            )
        y_ts_closed_form = closed_form_sim_ts["y_ts"]
        _, closed_form_sim_ts_underdamped = cfa_factory(
            ts_sim_closed_form, readout_dt=jnp.array(dt_readout), damping_regime="underdamped"
        )(y0, m, K, D, W, b)
        for key in closed_form_sim_ts_underdamped.keys():
            closed_form_sim_ts_underdamped[key] = closed_form_sim_ts_underdamped[key].reshape(
                (-1,) + closed_form_sim_ts_underdamped[key].shape[2:]
            )
        y_ts_closed_form_underdamped = closed_form_sim_ts_underdamped["y_ts"]

        benchmarking_data["low_precision_tsit_mse"].append(
            jnp.mean((y_ts_numerical_high_precision - y_ts_numerical_low_precision_tsit) ** 2)
        )
        benchmarking_data["low_precision_euler_mse"].append(
            jnp.mean((y_ts_numerical_high_precision - y_ts_numerical_low_precision_euler) ** 2)
        )
        benchmarking_data["cfa_mse"].append(
            jnp.mean((y_ts_numerical_high_precision - y_ts_closed_form) ** 2)
        )
        benchmarking_data["cfa_underdamped_mse"].append(
            jnp.mean((y_ts_numerical_high_precision - y_ts_closed_form_underdamped) ** 2)
        )

        if plot_solution is True:
            plt.figure()
            plt.plot(
                ts_readout,
                y_ts_numerical_high_precision[:, :num_units],
            )
            plt.xlabel("Time")
            plt.ylabel("Position")
            plt.legend()
            plt.grid()
            plt.box(True)
            plt.title("Coupled oscillator position")
            plt.show()

    for key, data in benchmarking_data.items():
        benchmarking_data[key] = jnp.array(data)
        """
        if key.split("_")[-1] == "mse":
            rmse_key = key.replace("mse", "rmse")
            benchmarking_data[rmse_key] = jnp.sqrt(benchmarking_data[key])
        """

    print(
        "Low precision Tsit5 RMSE mean:",
        jnp.mean(jnp.sqrt(benchmarking_data["low_precision_tsit_mse"])),
        "std:", jnp.std(jnp.sqrt(benchmarking_data["low_precision_tsit_mse"]))
    )
    print(
        "Low precision Euler RMSE mean:",
        jnp.mean(jnp.sqrt(benchmarking_data["low_precision_euler_mse"])),
        "std:", jnp.std(jnp.sqrt(benchmarking_data["low_precision_euler_mse"]))
    )
    print(
        "CFA RMSE mean:",
        jnp.mean(jnp.sqrt(benchmarking_data["cfa_mse"])),
        "std:", jnp.std(jnp.sqrt(benchmarking_data["cfa_mse"]))
    )
    print(
        "CFA underdamped RMSE mean:",
        jnp.mean(jnp.sqrt(benchmarking_data["cfa_underdamped_mse"])),
        "std:", jnp.std(jnp.sqrt(benchmarking_data["cfa_underdamped_mse"]))
    )


if __name__ == "__main__":
    print("Plotting the single rollout")
    plot_single_rollout()
    print("Benchmarking the simulation to real-time factor")
    benchmark_sim_to_real_time_factor()
    print("Benchmarking the integration error")
    benchmark_integration_error()

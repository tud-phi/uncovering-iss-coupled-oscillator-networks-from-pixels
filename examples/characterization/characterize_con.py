from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
from jax import Array, grad, jit, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)
figsize = (4.0, 2.8)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
lw = 2.0


# output directory
outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)


def con_ode_factory(K: Array, D: Array, W: Array, b: Array) -> Tuple[Callable, Callable]:

    def con_ode_fn(t: Array, y: Array, tau: Array) -> Array:
        x, x_d = jnp.split(y, 2)
        x_dd = tau - K @ x - D @ x_d - jnp.tanh(W @ x + b)
        y_d = jnp.concatenate([x_d, x_dd])
        return y_d

    def con_energy_fn(y: Array) -> Array:
        x, x_d = jnp.split(y, 2, axis=-1)
        U = 0.5 * jnp.sum(x.T @ K @ x)
        if jnp.linalg.det(W) > 0.0:
            U = U + jnp.sum(jnp.linalg.inv(W) @ jnp.log(jnp.cosh(W @ x + b)))
        T = 0.5 * jnp.sum(x_d.T @ x_d)
        return T + U

    return con_ode_fn, con_energy_fn


def simulate_ode(ode_fn: Callable, ts: Array, y0: Array, sim_dt: Array, tau: Optional[Array] = None, solver = Tsit5()) -> Dict[str, Array]:
    ode_term = ODETerm(ode_fn)
    if tau is None:
        tau = jnp.zeros((y0.shape[-1] // 2,))
    sol = diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=sim_dt,
        y0=y0,
        args=tau,
        saveat=SaveAt(ts=ts),
        max_steps=None,
    )
    sim_ts = dict(
        ts=ts,
        y_ts=sol.ys,
    )

    return sim_ts


def simulate_unstable_con():
    K = jnp.array([[-1.0]])
    D = jnp.array([[0.4]])
    W = jnp.array([[3.0]])
    b = jnp.array([0.0])

    ode_fn, energy_fn = con_ode_factory(K, D, W, b)
    ts = jnp.linspace(0.0, 4.0, 1000)
    sim_dt = jnp.array(1e-4)
    y0s = jnp.array([
        [-1.2, 0.0],
        [-1.0, 0.0],
        [-0.4, 0.0],
        [0.4, 0.0],
        [1.0, 0.0],
        [1.2, 0.0],
    ])

    # plot the trajectory
    fig, ax = plt.subplots(figsize=figsize)
    # ax.plot(sim_ts["ts"], sim_ts["y_ts"][:, 1], color=colors[1], label=r"$x_d$")
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_fn, ts, y0s[i], sim_dt=sim_dt)
        ax.plot(sim_ts["ts"], sim_ts["y_ts"][:, 0], linewidth=lw, color=colors[i], label=r"$x(0)=" + str(y0s[i, 0]) + "$")
    plt.box(True)
    plt.grid(True)
    ax.legend(loc="upper center", ncol=2)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outputs_dir / "unstable_con_time_series.pdf")
    plt.show()

    # create grid
    y_eqs = jnp.array([
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    ylim = jnp.array([[-3.0, 3.0], [-1.2, 1.2]])
    x_pts = jnp.linspace(ylim[0, 0], ylim[0, 1], 1000)
    x_d_pts = jnp.linspace(ylim[1, 0], ylim[1, 1], 1000)
    x_grid, x_d_grid = jnp.meshgrid(x_pts, x_d_pts)
    y_grid = jnp.stack([x_grid, x_d_grid], axis=-1)

    # evaluate total energy on grid
    E_grid = jax.vmap(jax.vmap(energy_fn))(y_grid)

    # evaluate the ODE on the grid
    y_d_grid = jax.vmap(
        jax.vmap(
            partial(
                ode_fn,
                jnp.array(0.0),
                tau=jnp.array([0.0]),
            )
        )
    )(y_grid)
    speed = jnp.sqrt(jnp.sum(y_d_grid ** 2, axis=-1))

    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(x_pts, x_d_pts, E_grid, levels=100)
    stream_lw = 2 * speed / speed.max()
    ax.streamplot(
        onp.array(x_pts),
        onp.array(x_d_pts),
        onp.array(y_d_grid[:, :, 0]),
        onp.array(y_d_grid[:, :, 1]),
        density=0.7,
        minlength=0.2,
        maxlength=100.0,
        linewidth=onp.array(stream_lw),
        color="k",
    )
    if y_eqs is not None:
        plt.plot(y_eqs[:, 0], y_eqs[:, 1], linestyle="None", marker="x", color="orange")
    plt.colorbar(cs, label="Total energy ($T + U$)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_xlim(ylim[0])
    ax.set_ylim(ylim[1])
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "unstable_con_phase_portrait.pdf")
    plt.show()


def simulate_unstable_coupling():
    K = jnp.array([[1.0, -1.4], [-1.4, 1.0]])
    D = jnp.diag(jnp.array([0.4, 0.4]))
    W = 3.0 * jnp.eye(2)
    b = jnp.zeros((2, ))

    ode_fn, energy_fn = con_ode_factory(K, D, W, b)
    ts = jnp.linspace(0.0, 7.0, 1000)
    sim_dt = jnp.array(1e-4)
    y0s = jnp.array([
        [1.0, 1.5, 0.0, 0.0],
        [3.5, 3.0, 0.0, 0.0],
    ])

    # plot the trajectory
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_fn, ts, y0s[i], sim_dt=sim_dt)
        ax.plot(sim_ts["ts"], sim_ts["y_ts"][:, 0],
                linewidth=lw, linestyle="--",
                color=colors[i],
                label=r"$x_1(0)=" + str(y0s[i, 0:2].tolist()) + "$")
        ax.plot(sim_ts["ts"], sim_ts["y_ts"][:, 1],
                linewidth=lw, linestyle=":",
                color=colors[i],
                label=r"$x_2(0)=" + str(y0s[i, 0:2].tolist()) + "$")
    plt.box(True)
    plt.grid(True)
    ax.legend(loc="upper center", ncol=2)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outputs_dir / "unstable_coupling_time_series.pdf")
    plt.show()

    # create grid
    x_eqs = jnp.array([
        [0.0, 0.0],
    ])
    xlim = 6 * jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    x1_pts = jnp.linspace(xlim[0, 0], xlim[0, 1], 500)
    x2_pts = jnp.linspace(xlim[1, 0], xlim[1, 1], 500)
    x1_grid, x2_grid = jnp.meshgrid(x1_pts, x2_pts)
    x_grid = jnp.stack([x1_grid, x2_grid], axis=-1)
    y_grid = jnp.concat([x_grid, jnp.zeros_like(x_grid)], axis=-1)

    # evaluate total energy on grid
    E_grid = jax.vmap(jax.vmap(energy_fn))(y_grid)

    # evaluate the ODE on the grid
    y_d_grid = jax.vmap(
        jax.vmap(
            partial(
                ode_fn,
                jnp.array(0.0),
                tau=jnp.zeros((2, )),
            )
        )
    )(y_grid)
    x_dd_grid = y_d_grid[..., 2:]
    speed = jnp.sqrt(jnp.sum(x_dd_grid ** 2, axis=-1))

    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(x1_pts, x2_pts, E_grid, levels=100)
    stream_lw = 2 * speed / speed.max()
    ax.streamplot(
        onp.array(x1_pts),
        onp.array(x2_pts),
        onp.array(x_dd_grid[:, :, 0]),
        onp.array(x_dd_grid[:, :, 1]),
        density=0.7,
        minlength=0.2,
        maxlength=100.0,
        linewidth=onp.array(stream_lw),
        color="k",
    )
    if x_eqs is not None:
        plt.plot(x_eqs[:, 0], x_eqs[:, 1], linestyle="None", marker="x", color="orange")
    plt.colorbar(cs, label="Potential energy $U$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(xlim[0])
    ax.set_ylim(xlim[1])
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "unstable_coupling_phase_portrait.pdf")
    plt.show()


def main():
    simulate_unstable_con()
    simulate_unstable_coupling()


if __name__ == "__main__":
    main()

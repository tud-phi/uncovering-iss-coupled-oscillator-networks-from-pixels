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


def con_ode_factory(
    K: Array, D: Array, W: Array, b: Array
) -> Tuple[Callable, Callable]:
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


def conw_ode_factory(
    Mw: Array, Kw: Array, Dw: Array, b: Array
) -> Tuple[Callable, Callable]:
    def conw_ode_fn(t: Array, yw: Array, tau: Array) -> Array:
        xw, xw_d = jnp.split(yw, 2)
        xw_dd = jnp.linalg.inv(Mw) @ (tau - Kw @ xw - Dw @ xw_d - jnp.tanh(xw + b))
        yw_d = jnp.concatenate([xw_d, xw_dd])
        return yw_d

    def conw_energy_fn(yw: Array) -> Array:
        xw, xw_d = jnp.split(yw, 2, axis=-1)
        U = 0.5 * jnp.sum(xw.T @ Kw @ xw) + jnp.sum(jnp.log(jnp.cosh(xw + b)))
        T = 0.5 * jnp.sum(xw_d.T @ Mw @ xw_d)
        return T + U

    return conw_ode_fn, conw_energy_fn


def pcon_ode_factory(
    K: Array, D: Array, W: Array, b: Array
) -> Tuple[Callable, Callable]:
    def pcon_ode_fn(t: Array, y: Array, tau: Array) -> Array:
        x, x_d = jnp.split(y, 2)
        x_dd = tau - K @ x - D @ x_d - jnp.linalg.inv(W) @ jnp.tanh(W @ x + b)
        # x_dd = tau - K @ x - D @ x_d - W.T @ jnp.tanh(W @ x + b)
        y_d = jnp.concatenate([x_d, x_dd])
        return y_d

    def pcon_energy_fn(y: Array) -> Array:
        x, x_d = jnp.split(y, 2, axis=-1)
        U = 0.5 * jnp.sum(x.T @ K @ x)
        # if jnp.linalg.det(W) > 0.0:
        #     U = U + jnp.sum(jnp.linalg.inv(W) @ jnp.log(jnp.cosh(W @ x + b)))
        T = 0.5 * jnp.sum(x_d.T @ x_d)
        return T + U

    return pcon_ode_fn, pcon_energy_fn


def simulate_ode(
    ode_fn: Callable,
    ts: Array,
    y0: Array,
    sim_dt: Array,
    tau: Optional[Array] = None,
    solver=Tsit5(),
) -> Dict[str, Array]:
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
    W = jnp.array([[0.0]])
    b = jnp.array([0.0])

    ode_fn, energy_fn = con_ode_factory(K, D, W, b)
    ts = jnp.linspace(0.0, 4.0, 1000)
    sim_dt = jnp.array(1e-4)
    y0s = jnp.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    # plot the trajectory
    fig, ax = plt.subplots(figsize=figsize)
    # ax.plot(sim_ts["ts"], sim_ts["y_ts"][:, 1], color=colors[1], label=r"$x_d$")
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_fn, ts, y0s[i], sim_dt=sim_dt)
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 0],
            linewidth=lw,
            color=colors[i],
            label=r"$x(0)=" + str(y0s[i, 0]) + "$",
        )
    plt.box(True)
    plt.grid(True)
    ax.legend(loc="upper center", ncol=2)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outputs_dir / "unstable_con_time_series.pdf")
    plt.show()

    # create grid
    y_eqs = jnp.array(
        [
            [0.0, 0.0],
        ]
    )
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
    speed = jnp.sqrt(jnp.sum(y_d_grid**2, axis=-1))

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
    b = jnp.zeros((2,))

    ode_fn, energy_fn = con_ode_factory(K, D, W, b)
    ts = jnp.linspace(0.0, 7.0, 1000)
    sim_dt = jnp.array(1e-4)
    y0s = jnp.array(
        [
            [1.0, 1.5, 0.0, 0.0],
            [3.5, 3.0, 0.0, 0.0],
        ]
    )

    # plot the trajectory
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_fn, ts, y0s[i], sim_dt=sim_dt)
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 0],
            linewidth=lw,
            linestyle="--",
            color=colors[i],
            label=r"$x_1(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 1],
            linewidth=lw,
            linestyle=":",
            color=colors[i],
            label=r"$x_2(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
    plt.box(True)
    plt.grid(True)
    ax.legend(loc="upper center", ncol=2)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outputs_dir / "unstable_coupling_time_series.pdf")
    plt.show()

    # create grid
    x_eqs = jnp.array(
        [
            [0.0, 0.0],
        ]
    )
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
                tau=jnp.zeros((2,)),
            )
        )
    )(y_grid)
    x_dd_grid = y_d_grid[..., 2:]
    speed = jnp.sqrt(jnp.sum(x_dd_grid**2, axis=-1))

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


def simulate_somehow_stable_conw():
    figsize = (6.0, 4.0)
    """
    K = jnp.array([[1.3, 1.0], [1.0, 1.0]])
    D = 0.1 * jnp.array([[1.5, 0.0], [0.0, 1.0]])
    W = jnp.array([[1.0, 2.0], [2.0, 5.0]])
    b = jnp.zeros((2,))
    """
    K = jnp.array([[4.0, 1.0], [1.0, 1.0]])
    D = jnp.array([[1.01, 1.0], [1.0, 1.0]])
    W = jnp.array([[1.0, 2.23], [2.23, 5.0]])
    b = jnp.array([0.0, 0.0])

    print("K:\n", K, "\nEigenvalues of K:", jnp.linalg.eigh(K).eigenvalues)
    print("D:\n", D, "Eigenvalues of D:", jnp.linalg.eigh(D).eigenvalues)
    print("W:\n", W, "Eigenvalues of W:", jnp.linalg.eigh(W).eigenvalues)

    # compute the matrices in the W coordinates
    Mw = jnp.linalg.inv(W)
    Kw = K @ W
    Dw = D @ W
    Kw_eigvals, Kw_eigvecs = jnp.linalg.eigh(Kw)
    Dw_eigvals, Dw_eigvecs = jnp.linalg.eigh(Dw)

    print("Kw:\n", Kw, "\nEigenvalues of Kw:", jnp.linalg.eigh(Kw).eigenvalues)
    print("Dw:\n", Dw, "\nEigenvalues of Dw:", jnp.linalg.eigh(Dw).eigenvalues)

    # create ode and energy functions in the original coordinates
    ode_original_coords_fn, energy_original_coords_fn = con_ode_factory(K, D, W, b)
    ode_w_coords_fn, energy_w_coords_fn = conw_ode_factory(Mw, Kw, Dw, b)
    ts = jnp.linspace(0.0, 1000.0, 1000)
    sim_dt = jnp.array(1e-4)
    y0s = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [100.0, -10.0, 0.0, 0.0],
        ]
    )

    # plot the trajectory in the original coordinates
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_original_coords_fn, ts, y0s[i], sim_dt=sim_dt)
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 0],
            linewidth=lw,
            linestyle="--",
            color=colors[i],
            label=r"$x_1(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 1],
            linewidth=lw,
            linestyle=":",
            color=colors[i],
            label=r"$x_2(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
    plt.box(True)
    plt.grid(True)
    ax.legend(ncol=2)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Oscillator position $x$")
    plt.tight_layout()
    plt.savefig(outputs_dir / "somehow_stable_conw_time_series_original_coords.pdf")
    plt.show()

    # plot the trajectory in the W coordinates
    tauw = Kw_eigvecs.T @ jnp.array([1000.0, 0.0])
    tauw = jnp.zeros((2,))
    # tauw = Dw_eigvecs.T @ jnp.array([1000.0, 0.0])
    print("tauw:", tauw)
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_w_coords_fn, ts, y0s[i], sim_dt=sim_dt, tau=tauw)
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 0],
            linewidth=lw,
            linestyle="--",
            color=colors[i],
            label=r"$x_{\mathrm{w},1}(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
        ax.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 1],
            linewidth=lw,
            linestyle=":",
            color=colors[i],
            label=r"$x_{\mathrm{w},2}(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
    plt.box(True)
    plt.grid(True)
    ax.legend(ncol=2)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Position in $\mathcal{W}$ coordinates $x_\mathrm{w}$")
    plt.tight_layout()
    plt.savefig(outputs_dir / "somehow_stable_conw_time_series_w_coords.pdf")
    plt.show()

    # create grid in the original coordinates
    x_eqs = jnp.array(
        [
            [0.0, 0.0],
        ]
    )
    xlim = 100 * jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    x1_pts = jnp.linspace(xlim[0, 0], xlim[0, 1], 500)
    x2_pts = jnp.linspace(xlim[1, 0], xlim[1, 1], 500)
    x1_grid, x2_grid = jnp.meshgrid(x1_pts, x2_pts)
    x_grid = jnp.stack([x1_grid, x2_grid], axis=-1)
    y_grid = jnp.concat([x_grid, jnp.zeros_like(x_grid)], axis=-1)
    # evaluate total energy on grid in the original coordinates
    E_grid = jax.vmap(jax.vmap(energy_original_coords_fn))(y_grid)
    # evaluate the ODE on the grid in the original coordinates
    y_d_grid = jax.vmap(
        jax.vmap(
            partial(
                ode_original_coords_fn,
                jnp.array(0.0),
                tau=jnp.zeros((2,)),
            )
        )
    )(y_grid)
    x_dd_grid = y_d_grid[..., 2:]
    speed = jnp.sqrt(jnp.sum(x_dd_grid**2, axis=-1))
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
    plt.savefig(outputs_dir / "somehow_stable_conw_phase_portrait_original_coords.pdf")
    plt.show()

    # create grid in the w coordinates
    x_eqs = jnp.array(
        [
            [0.0, 0.0],
        ]
    )
    xlim = 10 * jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    x1_pts = jnp.linspace(xlim[0, 0], xlim[0, 1], 1000)
    x2_pts = jnp.linspace(xlim[1, 0], xlim[1, 1], 1000)
    x1_grid, x2_grid = jnp.meshgrid(x1_pts, x2_pts)
    x_grid = jnp.stack([x1_grid, x2_grid], axis=-1)
    y_grid = jnp.concat([x_grid, jnp.zeros_like(x_grid)], axis=-1)
    # evaluate total energy on grid in the w coordinates
    E_grid = jax.vmap(jax.vmap(energy_w_coords_fn))(y_grid)
    # evaluate the ODE on the grid in the w coordinates
    y_d_grid = jax.vmap(
        jax.vmap(
            partial(
                ode_w_coords_fn,
                jnp.array(0.0),
                tau=jnp.zeros((2,)),
            )
        )
    )(y_grid)
    x_dd_grid = y_d_grid[..., 2:]
    speed = jnp.sqrt(jnp.sum(x_dd_grid**2, axis=-1))
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
    ax.set_xlabel(r"$x_{\mathrm{w},1}$")
    ax.set_ylabel(r"$x_{\mathrm{w},2}$")
    ax.set_xlim(xlim[0])
    ax.set_ylim(xlim[1])
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "somehow_stable_conw_phase_portrait_w_coords.pdf")
    plt.show()


def simulate_bistable_pcon():
    figsize = (6.0, 4.0)
    K = jnp.array([[5.0, -2.2], [-2.2, 1.0]])
    D = 0.2 * jnp.array([[1.0, 0.0], [0.0, 1.0]])
    W = jnp.array([[1.0, -2.2], [-2.2, 5.0]])
    b = jnp.array([0.0, 4.0])

    print("K:\n", K, "\nEigenvalues of K:", jnp.linalg.eigh(K).eigenvalues)
    print("D:\n", D, "Eigenvalues of D:", jnp.linalg.eigh(D).eigenvalues)
    print("W:\n", W, "Eigenvalues of W:", jnp.linalg.eigh(W).eigenvalues)

    ode_fn, energy_fn = pcon_ode_factory(K, D, W, b)
    ts = jnp.linspace(0.0, 80.0, 1000)
    sim_dt = jnp.array(2e-5)
    y0s = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [-0.5, -0.5, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
            [2.0, -2.0, 0.0, 0.0],
            [-2.0, 2.0, 0.0, 0.0],
            [-2.0, -2.0, 0.0, 0.0],
        ]
    )

    # plot the trajectory
    fig1, ax1 = plt.subplots(figsize=figsize)
    fig2, ax2 = plt.subplots(figsize=figsize)
    for i in range(y0s.shape[0]):
        sim_ts = simulate_ode(ode_fn, ts, y0s[i], sim_dt=sim_dt)
        print("yf:", sim_ts["y_ts"][-1])
        ax1.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 0],
            linewidth=lw,
            color=colors[i],
            label=r"$x(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
        ax2.plot(
            sim_ts["ts"],
            sim_ts["y_ts"][:, 1],
            linewidth=lw,
            color=colors[i],
            label=r"$x(0)=" + str(y0s[i, 0:2].tolist()) + "$",
        )
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(ncol=2)
    ax2.legend(ncol=2)
    ax1.set_xlabel(r"Time $t$ [s]")
    ax1.set_ylabel(r"1st oscillator position $x_1$")
    ax2.set_xlabel(r"Time $t$ [s]")
    ax2.set_ylabel(r"2nd oscillator position $x_2$")
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(outputs_dir / "bistable_pcon_time_series_x1.pdf")
    fig2.savefig(outputs_dir / "bistable_pcon_time_series_x2.pdf")
    plt.show()

    # create grid
    x_eqs = None
    x_eqs = jnp.array(
        [
            [-2.12488409e02, -4.74974002e02],
            [2.12488073e02, 4.74972926e02],
        ]
    )
    # xlim = 10 * jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    xlim = jnp.array([[-230.0, 230.0], [-490.0, 490.0]])
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
                tau=jnp.zeros((2,)),
            )
        )
    )(y_grid)
    x_dd_grid = y_d_grid[..., 2:]
    speed = jnp.sqrt(jnp.sum(x_dd_grid**2, axis=-1))

    fig, ax = plt.subplots(figsize=figsize)
    # cs = ax.contourf(x1_pts, x2_pts, E_grid, levels=100)
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
    # plt.colorbar(cs, label="Potential energy $U$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(xlim[0])
    ax.set_ylim(xlim[1])
    plt.box(True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "bistable_pcon_phase_portrait.pdf")
    plt.show()


def main():
    simulate_unstable_con()
    simulate_unstable_coupling()
    # simulate_somehow_stable_conw()
    simulate_bistable_pcon()


if __name__ == "__main__":
    main()

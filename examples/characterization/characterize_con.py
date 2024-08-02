from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
from jax import Array, grad, jit, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union


def con_ode_factory(K: Array, D: Array, W: Array, b: Array) -> Callable:

    def con_ode_fn(t: Array, y: Array, tau: Array) -> Array:
        x, x_d = jnp.split(y, 2)
        x_dd = tau - K @ x - D @ x_d - jnp.tanh(W @ x + b)
        y_d = jnp.concatenate([x_d, x_dd])
        return y_d

    return con_ode_fn


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
    )
    sim_ts = dict(
        ts=ts,
        y_ts=sol.ys,
    )

    return sim_ts

def simulate_multistable_con():
    ode_fn = con_ode_factory(K, D, W, b)
    ts = jnp.linspace(0.0, 10.0, 1000)
    y0 

    sim_ts = simulate_ode(ode_fn, ts, )


def main():
    simulate_multistable_con()


if __name__ == "__main__":
    main()

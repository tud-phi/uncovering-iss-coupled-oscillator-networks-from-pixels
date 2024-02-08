from clu import metrics as clu_metrics
from diffrax import AbstractSolver, diffeqsolve, Dopri5, ODETerm, SaveAt
from flax.core import FrozenDict
from flax.struct import dataclass
from flax import linen as nn  # Linen API
from functools import partial
from jax import Array, debug, jit, jvp, lax, random, vmap
import jax.numpy as jnp
from jsrm.systems.pendulum import normalize_joint_angles
from typing import Any, Callable, Dict, Optional, Tuple, Type

from src.losses.kld import kullback_leiber_divergence
from src.metrics import RootAverage
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.structs import TaskCallables


def task_factory(
    system_type: str,
    ts: Array,
    sim_dt: float,
    x0_min: Optional[Array] = None,
    x0_max: Optional[Array] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    dynamics_type: str = "node",  # ode, node or discrete
    nn_model: Optional[nn.Module] = None,
    ode_fn: Optional[Callable] = None,
    normalize_loss: bool = False,
    solver: AbstractSolver = Dopri5(),
    start_time_idx: int = 1,
    num_past_timesteps: int = 2,
) -> Tuple[TaskCallables, Type[clu_metrics.Collection]]:
    """
    Factory function for the task of learning a representation using first-principle dynamics while using the
    ground-truth velocity.
    Will return a TaskCallables object with the forward_fn, loss_fn, and compute_metrics functions.
    Args:
        system_type: the system type to create the task for. For example "pendulum".
        ts: time steps of the samples
        sim_dt: Time step used for simulation [s].
        x0_min: the minimal value for the initial state of the simulation
        x0_max: the maximal value for the initial state of the simulation
        loss_weights: the weights for the different loss terms
        dynamics_type: Dynamics type. One of ["ode", "node", "discrete"]
            ode: ODE dynamics map a state consisting of latent and their velocity to the state derivative.
            node: Neural ODE dynamics map (i.e., involving an neural network model) a state consisting of latent and their velocity to the state derivative.
            discrete: Discrete forward dynamics map the latents at the num_past_timesteps to the next latent.
        nn_model: the neural network model of the dynamics autoencoder. Should contain both the autoencoder and the neural ODE.
        ode_fn: (ground-truth) ODE function. Only mandatory if dynamics_type is "ode".
            It should have the following signature:
            ode_fn(t, x, tau) -> x_dot
        normalize_loss: whether to normalize the loss by the state bounds (i.e., x0_min and x0_max)
        solver: Diffrax solver to use for the simulation. Only use for the (neural) ODE.
        start_time_idx: the index of the time step to start the simulation at. Needs to be >=1 to enable the application
            of finite differences for the latent-space velocity.
        num_past_timesteps: the number of past timesteps to use for the discrete forward dynamics.
    Returns:
        task_callables: struct containing the functions for the learning task
        metrics_collection_cls: contains class for collecting metrics
    """
    # compute the dynamic rollout of the latent representation
    t0 = ts[start_time_idx]  # start time
    tf = ts[-1]  # end time
    # maximum of integrator steps
    max_int_steps = int((tf - t0) / sim_dt) + 1

    if loss_weights is None:
        loss_weights = {}
    loss_weights = dict(mse_q=1.0, mse_q_d=1.0) | loss_weights

    if dynamics_type == "discrete":
        assert (
            start_time_idx >= num_past_timesteps - 1
        ), "The start time idx needs to be >= num_past_timesteps - 1 to enable the passing of a sufficient number of inputs to the model."

    if normalize_loss is True:
        assert (
            x0_min is not None and x0_max is not None
        ), "x0_min and x0_max must be provided for normalizing the configuration loss"

    def assemble_input_fn(batch) -> Tuple[Array, Array]:
        # batch of images of shape batch_dim x time_dim x 2 * n_q
        x_bt = batch["x_ts"]
        if dynamics_type == "node":
            # return the state and the external torque
            x = x_bt[0, 0, ...]
            tau = batch["tau"][0, ...]
            return x, tau
        elif dynamics_type == "discrete":
            # return the state and the external torque
            x_past_ts = x_bt[0, :num_past_timesteps, ...]
            tau_ts = batch["tau"][0, None, ...].repeat(num_past_timesteps, axis=0)
            return x_past_ts, tau_ts
        else:
            raise ValueError(f"Unknown dynamics_type: {dynamics_type}")

    def forward_fn(
        batch: Dict[str, Array],
        nn_params: Optional[FrozenDict] = None,
        rng: Optional[random.KeyArray] = None,
        training: bool = False,
    ) -> Dict[str, Array]:
        # receive the state trajectory
        x_bt = batch["x_ts"]

        batch_size = batch["x_ts"].shape[0]
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        # construct batch of external torques of shape batch_dim x time_dim x n_tau
        tau_bt = jnp.expand_dims(batch["tau"], axis=1).repeat(x_bt.shape[1], axis=1)

        if dynamics_type in ["ode", "node"]:
            if dynamics_type == "node":
                # construct ode_fn and initiate ODE term
                def node_fn(t: Array, x: Array, tau: Array) -> Array:
                    x_d = nn_model.apply(
                        {"params": nn_params},
                        x,
                        tau,
                        method=nn_model.forward_dynamics,
                    )
                    return x_d

                ode_term = ODETerm(node_fn)
            else:
                ode_term = ODETerm(ode_fn)

            def ode_solve_fn(x0: Array, tau: Array):
                """
                Uses diffrax.diffeqsolve for solving the ode
                Arguments:
                    x0: initial state of shape (2*n_z, )
                    tau: external torques of shape (n_tau, )
                Returns:
                    sol: solution of the ode as a diffrax class
                """
                return diffeqsolve(
                    ode_term,
                    solver=solver,
                    t0=t0,  # initial time
                    t1=tf,  # final time
                    dt0=sim_dt,  # time step of integration
                    y0=x0,
                    args=tau,
                    saveat=SaveAt(ts=ts[start_time_idx:]),
                    max_steps=max_int_steps,
                )

            batched_ode_solve_fn = vmap(ode_solve_fn, in_axes=(0, 0))

            # initial configuration at the start time
            x_init_bt = x_bt[:, start_time_idx, ...]

            # simulate
            sol_bt = batched_ode_solve_fn(
                x_init_bt.astype(jnp.float64),  # initial state
                batch["tau"],  # external torques
            )

            # extract the rolled-out latent representations
            x_pred_bt = sol_bt.ys.astype(jnp.float32)
        elif dynamics_type == "discrete":
            # construct the input for the discrete forward dynamics
            x_past_bt = x_bt[:, start_time_idx - num_past_timesteps + 1 :]

            def autoregress_fn(_x_past_ts: Array, _tau_ts: Array) -> Array:
                """
                Autoregressive function for the discrete forward dynamics
                Arguments:
                    _x_past_ts: past latent representations of shape (num_past_timesteps, n_z)
                    _tau_ts: past & current external torques of shape (num_past_timesteps, n_tau)
                Returns:
                    _z_next: next latent representation of shape (batch_size, n_z)
                """

                _x_next = nn_model.apply(
                    {"params": nn_params},
                    _x_past_ts,
                    _tau_ts,
                    method=nn_model.forward_dynamics,
                )
                return _x_next

            def scan_fn(
                _carry: Dict[str, Array], _tau: Array
            ) -> Tuple[Dict[str, Array], Array]:
                """
                Function used as part of lax.scan rollout for the discrete forward dynamics
                Arguments:
                    _carry: carry state of the scan function
                    _tau: external torques of shape (n_tau, )
                """
                # append the current external torque to the past external torques
                _tau_ts = jnp.concatenate((_carry["tau_ts"], _tau[None, ...]), axis=0)

                # evaluate the autoregressive function
                _x_next = autoregress_fn(_carry["x_past_ts"], _tau_ts)

                # update the carry state
                _carry = dict(
                    x_past_ts=jnp.concatenate(
                        (_carry["x_past_ts"][1:], _x_next[None, ...]), axis=0
                    ),
                    tau_ts=_tau_ts[1:],
                )

                return _carry, _x_next

            def rollout_discrete_dynamics_fn(_x_ts: Array, _tau_ts: Array) -> Array:
                """
                Rollout function for the discrete forward dynamics
                Arguments:
                    _x_ts: latent representations of shape (time_dim, n_z)
                    _tau_ts: external torques of shape (time_dim, n_tau)
                Returns:
                    _z_dynamic_ts: next latent representation of shape (time_dim - start_time_idx, n_z)
                """
                carry_init = dict(
                    x_past_ts=_x_ts[
                        start_time_idx - num_past_timesteps + 1 : start_time_idx + 1
                    ],  # shape (num_past_timesteps, n_z)
                    tau_ts=_tau_ts[
                        start_time_idx - num_past_timesteps + 1 : start_time_idx
                    ],  # shape (num_past_timesteps - 1, n_tau)
                )
                carry_final, _x_ts = lax.scan(
                    scan_fn,
                    init=carry_init,
                    xs=_tau_ts[start_time_idx:],
                )

                return _x_ts

            x_pred_bt = vmap(
                rollout_discrete_dynamics_fn,
                in_axes=(0, 0),
                out_axes=0,
            )(x_past_bt, tau_bt.astype(jnp.float32))
        else:
            raise ValueError(f"Unknown dynamics_type: {dynamics_type}")

        preds = dict(
            x_ts=x_pred_bt,
        )

        return preds

    def loss_fn(
        batch: Dict[str, Array],
        nn_params: Optional[FrozenDict] = None,
        rng: Optional[random.KeyArray] = None,
        training: bool = False,
    ) -> Tuple[Array, Dict[str, Array]]:
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates
        preds = forward_fn(batch, nn_params, rng=rng, training=training)

        # compute the configuration error
        error_q = preds["x_ts"][..., :n_q] - batch["x_ts"][:, start_time_idx:, :n_q]
        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)
        # normalize the configuration error
        if normalize_loss:
            error_q = error_q / (x0_max[:n_q] - x0_min[:n_q])
        # compute the configuration MSE
        mse_q = jnp.mean(jnp.square(error_q))

        # compute the configuration velocity error
        error_q_d = preds["x_ts"][..., n_q:] - batch["x_ts"][:, start_time_idx:, n_q:]
        # compute the configuration velocity MSE
        mse_q_d = jnp.mean(jnp.square(error_q_d))

        # total loss
        loss = loss_weights["mse_q"] * mse_q + loss_weights["mse_q_d"] * mse_q_d

        return loss, preds

    def compute_metrics(
        batch: Dict[str, Array], preds: Dict[str, Array]
    ) -> Dict[str, Array]:
        n_q = batch["x_ts"].shape[-1] // 2  # number of generalized coordinates

        # compute the configuration error
        error_q = preds["x_ts"][..., :n_q] - batch["x_ts"][:, start_time_idx:, :n_q]
        # if necessary, normalize the joint angle error
        if system_type == "pendulum":
            error_q = normalize_joint_angles(error_q)

        # compute the configuration velocity error
        error_q_d = preds["x_ts"][..., n_q:] - batch["x_ts"][:, start_time_idx:, n_q:]

        if normalize_loss:
            error_q_norm = error_q / (x0_max[:n_q] - x0_min[:n_q])
            error_q_d_norm = error_q_d / (x0_max[n_q:] - x0_min[n_q:])
            batch_loss_dict = {
                "mse_q_norm": jnp.mean(jnp.square(error_q_norm)),
                "mse_q_d_norm": jnp.mean(jnp.square(error_q_d_norm)),
            }
        else:
            batch_loss_dict = {
                "mse_q": jnp.mean(jnp.square(error_q)),
                "mse_q_d": jnp.mean(jnp.square(error_q_d)),
            }

        return batch_loss_dict

    task_callables = TaskCallables(
        system_type, assemble_input_fn, forward_fn, loss_fn, compute_metrics
    )

    @dataclass  # <-- required for JAX transformations
    class MetricsCollection(clu_metrics.Collection):
        loss: clu_metrics.Average.from_output("loss")
        lr: clu_metrics.LastValue.from_output("lr")

        if normalize_loss:
            rmse_q_norm: RootAverage.from_output("mse_q_norm")
            rmse_q_d_norm: RootAverage.from_output("mse_q_d_norm")
        else:
            rmse_q: RootAverage.from_output("mse_q")
            rmse_q_d: RootAverage.from_output("mse_q_d")

    metrics_collection_cls = MetricsCollection
    return task_callables, metrics_collection_cls

from flax import linen as nn  # Linen API
from flax.training import orbax_utils
from orbax.checkpoint import Checkpointer, CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
import os
from typing import Optional, Union


from ciclo.callbacks import OptimizationMode
from ciclo.logging import Logs
from ciclo.loops.loop import (
    CallbackOutput,
    LoopCallbackBase,
    LoopState,
)
from ciclo.timetracking import Elapsed
from ciclo.types import S


class OrbaxCheckpoint(LoopCallbackBase[S]):
    def __init__(
            self,
            ckpt_dir: Union[str, os.PathLike],
            save_interval_steps: int = 1,
            max_to_keep: Optional[int] = None,
            keep_time_interval: Optional[int] = None,
            keep_period: Optional[int] = None,
            monitor: Optional[str] = None,
            mode: Union[str, OptimizationMode] = "min",
            use_orbax_save_args: bool = True
    ):
        if isinstance(mode, str):
            mode = OptimizationMode[mode]

        if mode not in OptimizationMode:
            raise ValueError(
                f"Invalid mode: {mode}, expected one of {list(OptimizationMode)}"
            )
        else:
            self.mode = mode

        self.ckpt_dir = ckpt_dir
        self.monitor = monitor
        self.use_orbax_save_args = use_orbax_save_args
        self.minimize = self.mode == OptimizationMode.min
        self._best: Optional[float] = None

        self.mngr_options = CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
            keep_time_interval=keep_time_interval,
            keep_period=keep_period,
            create=True
        )
        orbax_checkpointer = PyTreeCheckpointer()
        self.mngr = CheckpointManager(
            ckpt_dir, orbax_checkpointer, self.mngr_options
        )

    def __call__(
            self, elapsed: Elapsed, state: S, logs: Optional[Logs] = None
    ) -> None:
        save_checkpoint = True
        step_or_metric = elapsed.steps

        if self.monitor is not None:
            if logs is None:
                raise ValueError(
                    "checkpoint callback requires logs to monitor a metric"
                )
            if not isinstance(logs, Logs):
                logs = Logs(logs)

            try:
                value = logs.entry_value(self.monitor)
            except KeyError:
                raise ValueError(
                    f"Monitored value '{self.monitor}' not found in logs"
                )

            if (
                    self._best is None
                    or (self.minimize and value < self._best)
                    or (not self.minimize and value > self._best)
            ):
                self._best = value
                step_or_metric = (
                    value if self.mode == OptimizationMode.max else -value
                )
            else:
                save_checkpoint = False

        if save_checkpoint:
            save_kwargs = None
            if self.use_orbax_save_args:
                save_args = orbax_utils.save_args_from_target(state)
                save_kwargs = {'save_args': save_args}

            self.mngr.save(
                elapsed.steps, state, save_kwargs=save_kwargs
            )

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        self(loop_state.elapsed, loop_state.state, loop_state.accumulated_logs)
        return Logs(), loop_state.state

    def on_epoch_end(
            self, state, batch, elapsed, loop_state: LoopState[S]
    ) -> CallbackOutput[S]:
        return self.__loop_callback__(loop_state)

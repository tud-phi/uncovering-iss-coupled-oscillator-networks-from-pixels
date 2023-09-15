from typing import Any
from ciclo.logging import Logs
from ciclo.loops.loop import (
    CallbackOutput,
    LoopCallbackBase,
    LoopOutput,
    LoopState,
)
from ciclo.types import Batch, S
import optuna


class OptunaPruneCallback(LoopCallbackBase[S]):
    def __init__(self, trial: optuna.trial.Trial, metric_name: str = "loss_val"):
        super().__init__()
        self.trial = trial
        self.metric_name = metric_name

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        if self.metric_name in loop_state.logs["stateful_metrics"]:
            trial_step = int(loop_state.state.step.item())
            trial_value = float(
                loop_state.logs["stateful_metrics"][self.metric_name].item()
            )

            # report the current validation loss to optuna
            self.trial.report(trial_value, step=trial_step)

            # prune the trial if the current validation loss is too high
            if self.trial.should_prune():
                raise optuna.TrialPruned()

            # also prune if the loss is too high
            if loop_state.logs["stateful_metrics"]["loss_val"] > 1e6:
                raise optuna.TrialPruned()

        return Logs(), loop_state.state

    def on_train_batch_end(self, state, batch, elapsed, loop_state: LoopState[S]):
        return self.__loop_callback__(loop_state)

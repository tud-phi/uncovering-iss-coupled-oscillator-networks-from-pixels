import dill
import jax

jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
from pathlib import Path
from typing import Dict, List, Tuple


system_type = "pcc_ns-2"
sweep_folders = [
    Path("logs") / f"{system_type}_dynamics_autoencoder" / "2024-05-19_19-44-28",
    Path("logs") / f"{system_type}_dynamics_autoencoder" / "2024-05-20_11-05-49",
]
target_postfix = None
for sweep_folder in sweep_folders:
    assert sweep_folder.exists(), f"{sweep_folder} does not exist."
    if target_postfix is None:
        target_postfix = sweep_folder.name
    else:
        target_postfix += f"_{sweep_folder.name}"
target_folder = Path("logs") / f"{system_type}_dynamics_autoencoder" / target_postfix


def load_sweep_results(sweep_folder: Path) -> Dict:
    with open(sweep_folder / "sweep_results.dill", "rb") as file:
        sweep_results = dill.load(file)

    return sweep_results

def tree_concatenate(trees):
    """Takes a list of trees and concatenates every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((cat(a, a'), cat(b, b')), cat(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


if __name__ == "__main__":
    merged_sweep_results = None
    for sweep_folder in sweep_folders:
        sweep_results = load_sweep_results(sweep_folder)
        if merged_sweep_results is None:
            merged_sweep_results = sweep_results
        else:
            merged_sweep_results = tree_concatenate([merged_sweep_results, sweep_results])

    print("Merged test results:\n")
    for key, value in merged_sweep_results["test"].items():
        print(f"{key}:\n {value}")

    target_folder.mkdir(parents=True, exist_ok=True)
    with open(target_folder / "sweep_results.dill", "wb") as file:
        dill.dump(merged_sweep_results, file)
    print(f"Saved sweep results to {target_folder / 'sweep_results.dill'}")
import jax

jax.config.update("jax_enable_x64", True)  # double precision
import jsrm
from jsrm.systems import planar_hsa
from pathlib import Path

from hsa_planar_control.collocated_form import (
    check_integrability_assumption,
    symbolically_derive_collocated_form,
    mapping_into_collocated_form_factory,
)

num_segments = 1
num_rods_per_segment = 2


if __name__ == "__main__":
    assert (
        check_integrability_assumption(sym_exp_filepath) is True
    ), "Integrability assumption not satisfied"
    symbolically_derive_collocated_form(sym_exp_filepath)

    (
        _,
        _,
        _,
        _,
        _,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    map_into_collocated_form_fn, _ = mapping_into_collocated_form_factory(
        sym_exp_filepath, sys_helpers=sys_helpers, run_tests=True
    )
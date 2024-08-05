"""pendulum dataset."""

import dataclasses
import jax.numpy as jnp
from natsort import natsorted
import numpy as onp
from pathlib import Path
import shutil
import tensorflow_datasets as tfds
from typing import Optional


@dataclasses.dataclass
class PendulumDatasetConfig(tfds.core.BuilderConfig):
    state_dim: int = 2
    horizon_dim: int = 1
    img_size: tuple = (64, 64)
    num_links: int = 1
    num_simulations: int = 20000
    dt = 5e-2
    sim_dt = 2.5e-2
    seed: int = 0


class Pendulum(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for pendulum dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        PendulumDatasetConfig(
            name="single_pendulum_32x32px",
            description="Single pendulum dataset with images of size 32x32px.",
            state_dim=2,
            horizon_dim=11,
            img_size=(32, 32),
            num_links=1,
        ),
        PendulumDatasetConfig(
            name="single_pendulum_32x32px_h-101",
            description="Single pendulum dataset with images of size 32x32px and a horizon of 101 steps.",
            state_dim=2,
            horizon_dim=101,
            img_size=(32, 32),
            num_links=1,
        ),
        PendulumDatasetConfig(
            name="single_pendulum_64x64px",
            description="Single pendulum dataset with images of size 64x64px.",
            state_dim=2,
            horizon_dim=11,
            img_size=(64, 64),
            num_links=1,
        ),
        PendulumDatasetConfig(
            name="double_pendulum_32x32px",
            description="Double pendulum dataset with images of size 32x32px.",
            state_dim=4,
            horizon_dim=11,
            img_size=(32, 32),
            num_links=2,
        ),
        PendulumDatasetConfig(
            name="double_pendulum_64x64px",
            description="Double pendulum dataset with images of size 64x64px.",
            state_dim=4,
            horizon_dim=11,
            img_size=(64, 64),
            num_links=2,
        ),
    ]
    # pytype: enable=wrong-keyword-args

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": tfds.features.Scalar(dtype=jnp.int32),
                    "t_ts": tfds.features.Tensor(
                        shape=(self.builder_config.horizon_dim,),
                        dtype=onp.float64,
                    ),
                    "x_ts": tfds.features.Tensor(
                        shape=(
                            self.builder_config.horizon_dim,
                            self.builder_config.state_dim,
                        ),
                        dtype=onp.float64,
                    ),
                    "tau": tfds.features.Tensor(
                        shape=(self.builder_config.state_dim // 2,),
                        dtype=onp.float64,
                    ),
                    "rendering_ts": tfds.features.Sequence(
                        tfds.features.Image(
                            shape=(
                                self.builder_config.img_size[0],
                                self.builder_config.img_size[1],
                                3,
                            )
                        ),
                        length=self.builder_config.horizon_dim,
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(
                "rendering_ts",
                "rendering_ts",
            ),  # Set to `None` to disable
            homepage="https://github.com/tud-cor-sr/learning-representations-from-first-principle-dynamics",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        """Yields examples."""
        # lazy imports
        cv2 = tfds.core.lazy_imports.cv2
        jax = tfds.core.lazy_imports.jax
        jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
        jax.config.update("jax_enable_x64", True)  # double precision
        jnp = tfds.core.lazy_imports.jax.numpy
        plt = tfds.core.lazy_imports.matplotlib.pyplot
        # normal imports
        import diffrax
        from functools import partial
        import jsrm
        from jsrm.systems import pendulum

        from src.dataset_collection import collect_dataset
        from src.rendering import render_pendulum

        # Pseudo random number generator
        rng = jax.random.PRNGKey(seed=self.builder_config.seed)

        num_links = self.builder_config.num_links
        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"pendulum_nl-{num_links}.dill"
        )

        # set robot parameters
        robot_params = {
            "m": 10.0 * jnp.ones((num_links,)),
            "I": 3.0 * jnp.ones((num_links,)),
            "l": 2.0 * jnp.ones((num_links,)),
            "lc": 1.0 * jnp.ones((num_links,)),
            "g": jnp.array([0.0, -9.81]),
        }
        metadata = {"num_links": num_links}

        # initialize the system
        forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(
            sym_exp_filepath
        )

        n_q = self.builder_config.state_dim // 2  # number of configuration variables
        assert num_links == n_q, "Number of links must match half the state dimension."

        # initialize the rendering function
        # line thickness
        lw = 2
        metadata["rendering"] = {
            "width": self.builder_config.img_size[0],
            "height": self.builder_config.img_size[1],
            "line_thickness": lw,
        }
        rendering_fn = partial(
            render_pendulum,
            forward_kinematics_fn,
            robot_params,
            width=self.builder_config.img_size[0],
            height=self.builder_config.img_size[1],
            line_thickness=lw,
        )

        sample_q = 36 / 180 * jnp.pi * jnp.linspace(0.0, 1.0, num_links)
        # sample_q = jnp.array([0.0])
        sample_img = rendering_fn(sample_q)
        plt.figure(num="Sample rendering")
        plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        plt.title(f"q = {sample_q}")
        plt.show()

        # set initial conditions
        x0_min = jnp.zeros((2 * num_links,))
        x0_max = jnp.zeros((2 * num_links,))
        x0_min = x0_min.at[:num_links].set(-jnp.pi)
        x0_max = x0_max.at[:num_links].set(jnp.pi)
        # maximum magnitude of the initial joint velocity [rad/s]
        q_d0_max = jnp.pi * jnp.ones((num_links,))
        x0_min = x0_min.at[num_links:].set(-q_d0_max)
        x0_max = x0_max.at[num_links:].set(q_d0_max)

        # define maximum external torque
        tau_max = 2 * jnp.cumsum(
            jnp.max(robot_params["g"]) * robot_params["m"] * robot_params["lc"]
        )

        # collect the dataset
        yield from collect_dataset(
            ode_fn=jsrm.integration.ode_with_forcing_factory(
                dynamical_matrices_fn, robot_params
            ),
            rendering_fn=rendering_fn,
            rng=rng,
            num_simulations=self.builder_config.num_simulations,
            horizon_dim=self.builder_config.horizon_dim,
            dt=jnp.array(self.builder_config.dt),
            x0_min=x0_min,
            x0_max=x0_max,
            dataset_dir=str(self.data_path),
            solver=diffrax.Tsit5(),
            sim_dt=jnp.array(self.builder_config.sim_dt),
            system_params=robot_params,
            metadata=metadata,
            tau_max=tau_max,
        )

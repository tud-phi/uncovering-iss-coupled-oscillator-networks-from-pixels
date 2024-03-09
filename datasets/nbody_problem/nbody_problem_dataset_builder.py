"""planar_pcs dataset."""

import dataclasses
from jax import Array
import jax.numpy as jnp
from pathlib import Path
import tensorflow_datasets as tfds
from typing import List, Optional, Tuple


@dataclasses.dataclass
class NBodyProblemDatasetConfig(tfds.core.BuilderConfig):
    state_dim: int = 8
    horizon_dim: int = 101
    img_size: Tuple[int, int] = (32, 32)
    num_bodies: int = 2
    q0_max: Tuple = (0.7, 0.7, 0.7, 0.7)
    q_d0_max: Tuple = (1.0, 1.0, 1.0, 1.0)
    q_max: Tuple = (1.0, 1.0, 1.0, 1.0)
    q_d_max: Tuple = (10.0, 10.0, 10.0, 10.0)
    num_simulations: int = 10000
    dt: float = 3e-2
    sim_dt: float = 1e-4
    seed: int = 0


class NBodyProblem(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for planar pcs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        NBodyProblemDatasetConfig(
            name="nb-2_h-101_32x32px",
            description="Planar 2-body problem with images of size 32x32px and a horizon of 101 time steps.",
            num_bodies=2,
            state_dim=8,
            img_size=(32, 32),
        ),
        NBodyProblemDatasetConfig(
            name="nb-3_h-101_32x32px",
            description="Planar 3-body problem with images of size 32x32px and a horizon of 101 time steps.",
            num_bodies=3,
            state_dim=12,
            img_size=(32, 32),
            q0_max=(0.7, 0.7, 0.7, 0.7, 0.7, 0.7),
            q_d0_max=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            q_max=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            q_d_max=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        ),
    ]
    # pytype: enable=wrong-keyword-args

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset info."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": tfds.features.Scalar(dtype=jnp.int32),
                    "t_ts": tfds.features.Tensor(
                        shape=(self.builder_config.horizon_dim,),
                        dtype=jnp.float64,
                    ),
                    "x_ts": tfds.features.Tensor(
                        shape=(
                            self.builder_config.horizon_dim,
                            self.builder_config.state_dim,
                        ),
                        dtype=jnp.float64,
                    ),
                    "tau": tfds.features.Tensor(
                        shape=(self.builder_config.state_dim // 2,),
                        dtype=jnp.float64,
                    ),
                    "rendering_ts": tfds.features.Sequence(
                        tfds.features.Image(
                            shape=(
                                self.builder_config.img_size[1],
                                self.builder_config.img_size[0],
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
        """Returns SplitGenerators."""
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        """Yields examples."""
        # lazy imports
        cv2 = tfds.core.lazy_imports.cv2
        jax = tfds.core.lazy_imports.jax
        jax.config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
        jax.config.update("jax_enable_x64", True)  # double precision
        jnp = tfds.core.lazy_imports.jax.numpy
        plt = tfds.core.lazy_imports.matplotlib.pyplot
        # normal imports
        import diffrax
        from functools import partial
        from nbodyx.constants import G
        from nbodyx.ode import make_n_body_ode
        from nbodyx.rendering.opencv import render_n_body

        from src.dataset_collection import collect_dataset

        # Pseudo random number generator
        rng = jax.random.PRNGKey(seed=self.builder_config.seed)

        # set robot parameters
        num_bodies = self.builder_config.num_bodies
        system_params = {
            "body_masses": jnp.ones((num_bodies,)) / G,
        }
        metadata = {
            "num_bodies": num_bodies,
        }

        # set state bounds
        q_max = jnp.array(self.builder_config.q_max)
        q_d_max = jnp.array(self.builder_config.q_d_max)

        # define the colors
        colors = plt.cm.get_cmap("tab10").colors
        body_colors = (jnp.array(colors) * 255).astype(int)

        # initialize the rendering function
        metadata["rendering"] = {
            "width": self.builder_config.img_size[0],
            "height": self.builder_config.img_size[1],
            "body_radii": 0.04
            * min(self.builder_config.img_size)
            * jnp.ones((num_bodies,)),
            "body_colors": body_colors,
        }
        rendering_fn = partial(
            render_n_body,
            width=self.builder_config.img_size[0],
            height=self.builder_config.img_size[1],
            x_min=jnp.min(-q_max.reshape(num_bodies, -1), axis=0),
            x_max=jnp.max(q_max.reshape(num_bodies, -1), axis=0),
            body_radii=metadata["rendering"]["body_radii"],
            body_colors=metadata["rendering"]["body_colors"],
        )

        # set initial conditions and state bounds
        x0_min = jnp.concatenate(
            [
                -jnp.array(self.builder_config.q0_max),
                -jnp.array(self.builder_config.q_d0_max),
            ],
            axis=0,
        )
        x0_max = jnp.concatenate(
            [
                jnp.array(self.builder_config.q0_max),
                jnp.array(self.builder_config.q_d0_max),
            ],
            axis=0,
        )
        x_min = jnp.concatenate(
            [
                -jnp.array(self.builder_config.q_max),
                -jnp.array(self.builder_config.q_d_max),
            ],
            axis=0,
        )
        x_max = jnp.concatenate(
            [
                jnp.array(self.builder_config.q_max),
                jnp.array(self.builder_config.q_d_max),
            ],
            axis=0,
        )

        sample_q = jnp.linspace(0.0, 1.0, 2 * num_bodies) * jnp.array(
            self.builder_config.q0_max
        )
        sample_img = rendering_fn(sample_q)
        plt.figure(num="Sample rendering")
        plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        plt.title(f"q = {sample_q}")
        plt.show()

        # define maximum torque as zero
        tau_max = jnp.zeros((self.builder_config.state_dim // 2,))

        # collect the dataset
        yield from collect_dataset(
            ode_fn=make_n_body_ode(system_params["body_masses"]),
            rendering_fn=rendering_fn,
            rng=rng,
            num_simulations=self.builder_config.num_simulations,
            horizon_dim=self.builder_config.horizon_dim,
            dt=jnp.array(self.builder_config.dt),
            x0_min=x0_min,
            x0_max=x0_max,
            dataset_dir=str(self.data_path),
            solver=diffrax.Dopri5(),
            sim_dt=jnp.array(self.builder_config.sim_dt),
            system_params=system_params,
            metadata=metadata,
            x0_sampling_dist="uniform",
            tau_max=tau_max,
            enforce_state_bounds=True,
            x_min=x_min,
            x_max=x_max,
            save_raw_data=False,
            animate_trajectory=False,
        )

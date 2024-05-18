"""planar_pcs dataset."""

import dataclasses
from jax import Array
import jax.numpy as jnp
from pathlib import Path
import tensorflow_datasets as tfds
from typing import List, Optional, Tuple


@dataclasses.dataclass
class PlanarPcsDatasetConfig(tfds.core.BuilderConfig):
    state_dim: Optional[int] = None
    horizon_dim: int = 11
    img_size: Tuple[int, int] = (64, 64)
    origin_uv: Tuple[int, int] = (32, 8)
    num_segments: int = 1
    strain_selector: Optional[Tuple] = (None,)
    q0_max: Tuple = (10 * jnp.pi, 0.05, 0.1)
    q_d0_max: Tuple = (10 * jnp.pi, 0.05, 0.1)
    num_simulations: int = 20000
    dt: float = 2e-2
    sim_dt: float = 1e-4
    seed: int = 0


class PlanarPcs(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for planar pcs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        PlanarPcsDatasetConfig(
            name="cc_64x64px",
            description="Planar constant curvature continuum robot with images of size 64x64px.",
            state_dim=2,
            horizon_dim=11,
            num_segments=1,
            strain_selector=(True, False, False),
            q0_max=(10 * jnp.pi,),
            q_d0_max=(0.2 * jnp.pi,),
        ),
        PlanarPcsDatasetConfig(
            name="cs_64x64px",
            description="Planar constant strain continuum robot with images of size 64x64px.",
            state_dim=6,
            num_segments=1,
        ),
        PlanarPcsDatasetConfig(
            name="cs_32x32px_h-101",
            description="Planar constant strain continuum robot with images of size 32x32px.",
            state_dim=6,
            horizon_dim=101,
            img_size=(32, 32),
            origin_uv=(16, 4),
            num_segments=1,
            strain_selector=(True, True, True),
            q0_max=(5 * jnp.pi, 0.2, 0.2),
            q_d0_max=(5 * jnp.pi, 0.2, 0.2),
            num_simulations=15000,
        ),
        PlanarPcsDatasetConfig(
            name="pcc_ns-2_32x32px",
            description="Planar two segment piecewise constant curvature continuum robot with images of size 32x32px.",
            state_dim=4,
            img_size=(32, 32),
            origin_uv=(16, 4),
            num_segments=2,
            strain_selector=(True, False, False, True, False, False),
            q0_max=(5 * jnp.pi, 5 * jnp.pi),
            q_d0_max=(5 * jnp.pi, 5 * jnp.pi),
            sim_dt=1e-4,
        ),
        PlanarPcsDatasetConfig(
            name="pcc_ns-2_32x32px_h-101",
            description="Planar two segment piecewise constant curvature continuum robot with images of size 32x32px and a horizon of 101 steps.",
            state_dim=4,
            horizon_dim=101,
            img_size=(32, 32),
            origin_uv=(16, 4),
            num_segments=2,
            strain_selector=(True, False, False, True, False, False),
            q0_max=(5 * jnp.pi, 5 * jnp.pi),
            q_d0_max=(5 * jnp.pi, 5 * jnp.pi),
            num_simulations=10000,
            sim_dt=1e-4,
        ),
        PlanarPcsDatasetConfig(
            name="pcc_ns-2_64x64px",
            description="Planar two segment piecewise constant curvature continuum robot with images of size 64x64px.",
            state_dim=4,
            num_segments=2,
            strain_selector=(True, False, False, True, False, False),
            q0_max=(5 * jnp.pi, 5 * jnp.pi),
            q_d0_max=(5 * jnp.pi, 5 * jnp.pi),
        ),
        PlanarPcsDatasetConfig(
            name="pcc_ns-3_32x32px_h-101",
            description="Planar three segment piecewise constant curvature continuum robot with images of size 32x32px and a horizon of 101 steps.",
            state_dim=6,
            horizon_dim=101,
            img_size=(32, 32),
            origin_uv=(16, 4),
            num_segments=3,
            strain_selector=(True, False, False, True, False, False, True, False, False),
            q0_max=(5 * jnp.pi, 5 * jnp.pi, 5 * jnp.pi),
            q_d0_max=(5 * jnp.pi, 5 * jnp.pi, 5 * jnp.pi),
            num_simulations=14000,
            sim_dt=1e-4,
        ),
        PlanarPcsDatasetConfig(
            name="pcc_ns-3_64x64px",
            description="Planar three segment piecewise constant curvature continuum robot with images of size 64x64px.",
            state_dim=6,
            num_segments=3,
            strain_selector=(
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
            ),
            q0_max=(3.33 * jnp.pi, 3.33 * jnp.pi, 3.33 * jnp.pi),
            q_d0_max=(3.33 * jnp.pi, 3.33 * jnp.pi, 3.33 * jnp.pi),
        ),
        PlanarPcsDatasetConfig(
            name="pcc_ns-4_32x32px_h-101",
            description="Planar four segment piecewise constant curvature continuum robot with images of size 32x32px and a horizon of 101 steps.",
            state_dim=8,
            horizon_dim=101,
            img_size=(32, 32),
            origin_uv=(16, 4),
            num_segments=4,
            strain_selector=(True, False, False, True, False, False, True, False, False, True, False, False),
            q0_max=(5 * jnp.pi, 5 * jnp.pi, 5 * jnp.pi, 5 * jnp.pi),
            q_d0_max=(5 * jnp.pi, 5 * jnp.pi, 5 * jnp.pi, 5 * jnp.pi),
            num_simulations=14000,
            sim_dt=1e-4,
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
        jax.config.update("jax_platforms", "cpu")  # set default device to 'cpu'
        jax.config.update("jax_enable_x64", True)  # double precision
        jnp = tfds.core.lazy_imports.jax.numpy
        plt = tfds.core.lazy_imports.matplotlib.pyplot
        # normal imports
        import diffrax
        from functools import partial
        import jsrm
        from jsrm.systems import planar_pcs

        from src.dataset_collection import collect_dataset
        from src.rendering import render_planar_pcs

        # Pseudo random number generator
        rng = jax.random.PRNGKey(seed=self.builder_config.seed)

        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_pcs_ns-{self.builder_config.num_segments}.dill"
        )

        # set robot parameters
        strain_selector = jnp.array(self.builder_config.strain_selector)
        rho = 600 * jnp.ones((self.builder_config.num_segments,))  # [kg/m^3]
        # damping matrix
        D = 1e-5 * jnp.diag(
            jnp.repeat(
                jnp.array([[1e0, 1e3, 1e3]]), self.builder_config.num_segments, axis=0
            ).flatten(),
        )
        robot_params = {
            "th0": jnp.array(jnp.pi),  # initial orientation angle [rad]
            "l": 1e-1 * jnp.ones((self.builder_config.num_segments,)),
            "r": 1e-2 * jnp.ones((self.builder_config.num_segments,)),
            "rho": rho,
            "g": jnp.array([0.0, -9.81]),
            # Elastic modulus [Pa]
            "E": 2e4 * jnp.ones((self.builder_config.num_segments,)),
            # Shear modulus [Pa]
            "G": 1e4 * jnp.ones((self.builder_config.num_segments,)),
            "D": D,
        }
        metadata = {
            "num_segments": self.builder_config.num_segments,
            "strain_selector": strain_selector,
        }

        # initialize the system
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(
            sym_exp_filepath, strain_selector
        )

        n_q = self.builder_config.state_dim // 2  # number of configuration variables
        # check that the state dimension is correct
        assert (
            n_q == strain_basis.shape[1]
        ), "Provided state dimension does not match the strain selector / num of segments!"
        assert (
            n_q == len(self.builder_config.q0_max)
        ), "Provided state dimension does not match the number of provided q0_max values!"
        assert (
            n_q == len(self.builder_config.q_d0_max)
        ), "Provided state dimension does not match the number of provided q_d0_max values!"

        # initialize the rendering function
        # the line thickness is calibrated for 64x64px images
        lw = int(6 / 64 * jnp.mean(jnp.array(self.builder_config.img_size)))
        metadata["rendering"] = {
            "width": self.builder_config.img_size[0],
            "height": self.builder_config.img_size[1],
            "origin_uv": self.builder_config.origin_uv,
            "line_thickness": lw,
        }
        rendering_fn = partial(
            render_planar_pcs,
            forward_kinematics_fn,
            robot_params,
            width=self.builder_config.img_size[0],
            height=self.builder_config.img_size[1],
            origin_uv=self.builder_config.origin_uv,
            line_thickness=lw,
        )

        sample_q = jnp.array(self.builder_config.q0_max)
        # sample_q = jnp.array([0.0])
        sample_img = rendering_fn(sample_q)
        plt.figure(num="Sample rendering")
        plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        plt.title(f"q = {sample_q}")
        plt.show()

        # set initial conditions
        q0_max = jnp.array(self.builder_config.q0_max)
        q_d0_max = jnp.array(self.builder_config.q_d0_max)
        x0_min = jnp.concatenate([-q0_max, -q_d0_max], axis=0)
        x0_max = jnp.concatenate([q0_max, q_d0_max], axis=0)

        # define maximum torque as some scaling of the steady-state torques acting at (q0_max, q_d0_max)
        B, C, G, K, D, alpha = dynamical_matrices_fn(
            robot_params, q0_max, jnp.zeros_like(q_d0_max)
        )
        tau_max = 1.0 * jnp.abs(G + K)
        # tau_max = 0.3 * jnp.abs(G + K)
        print(f"tau_max = {tau_max}")

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
            solver=diffrax.Dopri5(),
            sim_dt=jnp.array(self.builder_config.sim_dt),
            system_params=robot_params,
            metadata=metadata,
            x0_sampling_dist="uniform",
            tau_max=tau_max,
            save_raw_data=False,
        )

"""planar_pcs dataset."""
import dataclasses
from jax import Array
import jax.numpy as jnp
from natsort import natsorted
from pathlib import Path
import tensorflow_datasets as tfds
from typing import List, Optional, Tuple


@dataclasses.dataclass
class PlanarPcsDatasetConfig(tfds.core.BuilderConfig):
    path: Optional[Path] = None
    state_dim: Optional[int] = None
    horizon_dim: int = 11
    img_size: tuple = (128, 128)
    num_segments: int = 1
    strain_selector: Optional[Tuple] = (None,)
    q_max: Tuple = (3 * jnp.pi, 0.02, 0.1)
    q_d_max: Tuple = (3 * jnp.pi, 0.02, 0.1)
    num_simulations: int = 20000
    dt: float = 5e-2
    sim_dt: float = 5e-3
    seed: int = 0


class PlanarPcs(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mechanical_system dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        PlanarPcsDatasetConfig(
            name="cc_128x128px",
            description="Planar constant curvature continuum robot with images of size 128x128px.",
            path=Path("data") / "raw_datasets" / "cc_64x64px",
            state_dim=2,
            horizon_dim=11,
            img_size=(128, 128),
            num_segments=1,
            strain_selector=(True, False, False),
            q_max=(3 * jnp.pi,),
            q_d_max=(3 * jnp.pi,),
        ),
        PlanarPcsDatasetConfig(
            name="cs_64x64px",
            description="Planar constant strain continuum robot with images of size 128x128px.",
            path=Path("data") / "raw_datasets" / "cc_128x128px",
            state_dim=6,
            horizon_dim=11,
            img_size=(128, 128),
            num_segments=1,
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
                        dtype=jnp.float64,
                    ),
                    "x_ts": tfds.features.Tensor(
                        shape=(
                            self.builder_config.horizon_dim,
                            self.builder_config.state_dim,
                        ),
                        dtype=jnp.float64,
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
        """Returns SplitGenerators."""
        return {
            "train": self._generate_examples(self.builder_config.path),
        }

    def _generate_examples(self, path: Path):
        """Yields examples."""
        # lazy imports
        cv2 = tfds.core.lazy_imports.cv2
        jax_config = tfds.core.lazy_imports.jax.config
        jax_config.update("jax_platform_name", "cpu")  # set default device to 'cpu'
        jax_config.update("jax_enable_x64", True)  # double precision
        jax = tfds.core.lazy_imports.jax
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
        rho = 1070 * jnp.ones(
            (self.builder_config.num_segments,)
        )  # Volumetric density of Dragon Skin 20 [kg/m^3]
        # damping matrix
        D = 1e-5 * jnp.diag(
            jnp.repeat(
                jnp.array([1e0, 1e3, 1e3]), self.builder_config.num_segments, axis=0
            ),
        )
        robot_params = {
            "th0": jnp.array(jnp.pi),  # initial orientation angle [rad]
            "l": 1e-1 * jnp.ones((self.builder_config.num_segments,)),
            "r": 2e-2 * jnp.ones((self.builder_config.num_segments,)),
            "rho": rho,
            "g": jnp.array([0.0, -9.81]),
            "E": 1e4
            * jnp.ones((self.builder_config.num_segments,)),  # Elastic modulus [Pa]
            "G": 1e3
            * jnp.ones((self.builder_config.num_segments,)),  # Shear modulus [Pa]
            "D": D,
        }

        # initialize the system
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
            sym_exp_filepath, strain_selector
        )

        n_q = self.builder_config.state_dim // 2  # number of configuration variables
        # check that the state dimension is correct
        assert (
            n_q == strain_basis.shape[1]
        ), "Provided state dimension does not match the strain selector / num of segments!"
        assert n_q == len(
            self.builder_config.q_max
        ), "Provided state dimension does not match the number of provided q_max values!"
        assert n_q == len(
            self.builder_config.q_d_max
        ), "Provided state dimension does not match the number of provided q_d_max values!"

        # initialize the rendering function
        rendering_fn = partial(
            render_planar_pcs,
            forward_kinematics_fn,
            robot_params,
            width=self.builder_config.img_size[0],
            height=self.builder_config.img_size[1],
            line_thickness=2,
        )

        sample_q = jnp.array(self.builder_config.q_max)
        sample_img = rendering_fn(sample_q)
        plt.figure(num="Sample rendering")
        plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        plt.title(f"q = {sample_q}")
        plt.show()

        # set initial conditions
        q_max = jnp.array(self.builder_config.q_max)
        q_d_max = jnp.array(self.builder_config.q_d_max)
        state_init_min = jnp.concatenate([-q_max, -q_d_max], axis=0)
        state_init_max = jnp.concatenate([q_max, q_d_max], axis=0)

        # set initial / torque conditions
        tau = jnp.zeros((n_q,))

        # collect the dataset
        yield from collect_dataset(
            ode_fn=jsrm.integration.ode_factory(
                dynamical_matrices_fn, robot_params, tau
            ),
            rendering_fn=rendering_fn,
            rng=rng,
            num_simulations=self.builder_config.num_simulations,
            horizon_dim=self.builder_config.horizon_dim,
            dt=jnp.array(self.builder_config.dt),
            state_init_min=state_init_min,
            state_init_max=state_init_max,
            dataset_dir=str(path),
            solver=diffrax.Dopri5(),
            sim_dt=jnp.array(self.builder_config.sim_dt),
            system_params=robot_params,
        )

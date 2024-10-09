"""reaction_diffusion dataset."""

import dataclasses
import numpy as onp
from pathlib import Path
import tensorflow_datasets as tfds
from typing import List, Optional, Tuple


@dataclasses.dataclass
class ReactionDiffusionDatasetConfig(tfds.core.BuilderConfig):
    # these parameters need to agree with the parameters chosen in the
    # examples/datasets/reaction_diffusion/generate_reaction_diffusion_dataset.m MATLAB script
    horizon_dim: int = 101
    img_size: Tuple[int, int] = (32, 32)
    origin_uv: Tuple[int, int] = (0, 0)
    dt: float = 5e-2


class ReactionDiffusion(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for planar pcs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        ReactionDiffusionDatasetConfig(
            name="reaction_diffusion_default",
            description="The default configuration for the reactor diffusion dataset.",
        ),
    ]
    # pytype: enable=wrong-keyword-args

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset info."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": tfds.features.Scalar(dtype=onp.int32),
                    "t_ts": tfds.features.Tensor(
                        shape=(self.builder_config.horizon_dim,),
                        dtype=onp.float64,
                    ),
                    "tau": tfds.features.Tensor(
                        shape=(2,),
                        dtype=onp.float64,
                    ),
                    "rendering_ts": tfds.features.Sequence(
                        tfds.features.Image(
                            shape=(
                                self.builder_config.img_size[1],
                                self.builder_config.img_size[0],
                                2,
                            )
                        ),
                        length=self.builder_config.horizon_dim,
                    ),
                    "rendering_d_ts": tfds.features.Sequence(
                        tfds.features.Image(
                            shape=(
                                self.builder_config.img_size[1],
                                self.builder_config.img_size[0],
                                2,
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
        # normal imports
        import dill
        import scipy.io as sio

        # dataset directory
        dataset_path = Path(self.data_dir)
        # (re)create the directory
        dataset_path.mkdir(parents=True, exist_ok=True)

        # load the data
        data = sio.loadmat(
            str(Path(__file__).parent / "reaction_diffusion.mat")
        )
        # extract the data
        ts = data["t"].squeeze()
        # shape of vector fields: (img_width, img_height, num_timesteps)
        u_ts = data["uf"]
        v_ts = data["vf"]
        u_d_ts = data["duf"]
        v_d_ts = data["dvf"]
        # transpose to have the time dimension first
        u_ts = onp.transpose(u_ts, (2, 0, 1))
        v_ts = onp.transpose(v_ts, (2, 0, 1))
        u_d_ts = onp.transpose(u_d_ts, (2, 0, 1))
        v_d_ts = onp.transpose(v_d_ts, (2, 0, 1))

        # generate the rendering by concatenating the u and v channels
        rendering_ts = onp.stack([u_ts, v_ts], axis=-1)
        rendering_d_ts = onp.stack([u_d_ts, v_d_ts], axis=-1)

        # normalize the images to be in the range [0, 255] and cast to uint8
        img_min, img_max = rendering_ts.min(), rendering_ts.max()
        rendering_ts = ((rendering_ts - img_min) / (img_max - img_min) * 255).astype(onp.uint8)
        rendering_d_ts = (rendering_d_ts / (img_max - img_min) * 255).astype(onp.uint8)

        # determine the number of samples in the dataset
        horizon_dim = self.builder_config.horizon_dim
        num_rollouts = ts.shape[0] // horizon_dim

        # remove any samples that are not full rollouts
        ts = ts[:num_rollouts * horizon_dim]
        rendering_ts = rendering_ts[:num_rollouts * horizon_dim]
        rendering_d_ts = rendering_d_ts[:num_rollouts * horizon_dim]

        # reshape to add the horizon dimension
        ts_rls = onp.reshape(ts, (num_rollouts, horizon_dim))
        rendering_ts_rls = onp.reshape(rendering_ts, (num_rollouts, horizon_dim, *rendering_ts.shape[1:]))
        rendering_d_ts_rls = onp.reshape(rendering_d_ts, (num_rollouts, horizon_dim, *rendering_d_ts.shape[1:]))
        # virtual torque that is zero for compatibility with the rest of the datasets
        tau_rls = onp.zeros((num_rollouts, 2), dtype=onp.float64)

        # recalibrate the time to start from zero
        ts_rls = ts_rls - ts_rls[:, 0][:, None]

        # define and save the metadata
        metadata = dict(
            dt=self.builder_config.dt,
            ts=ts_rls[0],
            solver_class="Tsit5",
            tau_max=tau_rls.max(axis=0),
            system_params=dict(
                d1=0.1,
                d2=0.1,
                beta=1.0,
            ),
            rendering=dict(
                width=self.builder_config.img_size[0],
                height=self.builder_config.img_size[1],
                origin_uv=self.builder_config.origin_uv,
                img_min=img_min,
                img_max=img_max,
            )
        )
        print("Metadata:\n", metadata)
        # save the metadata in the `dataset_dir`
        with open(str(dataset_path / "metadata.pkl"), "wb") as f:
            dill.dump(metadata, f)

        # iterate over the rollouts
        for rollout_idx in range(num_rollouts):
            # merge labels with image and id
            sample = {
                "id": rollout_idx,
                "t_ts": ts_rls[rollout_idx],
                "tau": tau_rls[rollout_idx],
                "rendering_ts": rendering_ts_rls[rollout_idx],
                "rendering_d_ts": rendering_d_ts_rls[rollout_idx],
            }

            yield rollout_idx, sample

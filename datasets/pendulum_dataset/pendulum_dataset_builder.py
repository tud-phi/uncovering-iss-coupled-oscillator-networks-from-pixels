"""pendulum_dataset dataset."""
import dataclasses
import jax.numpy as jnp
from pathlib import Path
import tensorflow_datasets as tfds


@dataclasses.dataclass
class DatasetConfig(tfds.core.BuilderConfig):
    path: Path = None
    state_dim: int = None
    horizon_dim: int = 1
    img_size: tuple = (32, 32)

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for pendulum_dataset dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        DatasetConfig(
            name='test',
            description='Test ...',
            path=Path("data/test"),
            state_dim=2,
            horizon_dim=10,
            img_size=(32, 32)
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
                    # "img_ts": tfds.features.Sequence(
                    #     tfds.features.Image(shape=(
                    #         self.builder_config.img_size[0],
                    #         self.builder_config.img_size[1],
                    #         3
                    #     ))
                    # ),
                    "x_ts": tfds.features.Tensor(
                        shape=(self.builder_config.horizon_dim, self.builder_config.state_dim),
                        dtype=jnp.float64
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("rendering", "state"),  # Set to `None` to disable
            homepage="https://github.com/tud-cor-sr/learning-representations-from-first-principle-dynamics",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            "train": self._generate_examples(self.builder_config.path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # lazy imports
        cv2 = tfds.core.lazy_imports.cv2

        print("test")
        print("path", path.resolve())
        for sim_labels in path.glob("*.npz"):
            filename = sim_labels.stem
            print("filename", filename)
            sim_idx = int(filename.lstrip("sim-").rstrip("_labels"))
            print("sim_idx", sim_idx)

            labels = jnp.load(sim_labels)
            print("state", labels["x_ts"].shape)

            # f = tfds.core.lazy_imports.cv2.imread(str(sim_labels).replace(".npz", ".png"))

            # for sim_labels in path.glob("*.npz"):

            yield sim_idx, {
                # "rendering": f,
                "id": sim_idx,
                "x_ts": labels["x_ts"],
            }

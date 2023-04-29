from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jax_resnet.common import ConvBlock, ModuleDef
from jax_resnet.resnet import ResNetBlock, ResNetBottleneckBlock, ResNetStem, STAGE_SIZES
from jax_resnet.splat import SplAtConv2d


class ResNetEncoder(nn.Module):
    """
    A ResNet encoder.
    """
    block_cls: ModuleDef
    stage_sizes: Sequence[int]
    latent_dim: int
    hidden_sizes: Sequence[int] = (64, 128, 256, 512)
    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)
    conv_block_cls: ModuleDef = ConvBlock
    stem_cls: ModuleDef = ResNetStem
    pool_fn: Callable = partial(nn.max_pool,
                                window_shape=(3, 3),
                                strides=(2, 2),
                                padding=((1, 1), (1, 1)))

    def setup(self):
        conv_block_cls = partial(self.conv_block_cls, conv_cls=self.conv_cls, norm_cls=self.norm_cls)
        stem_cls = partial(self.stem_cls, conv_block_cls=conv_block_cls)
        block_cls = partial(self.block_cls, conv_block_cls=conv_block_cls)

        layers = [stem_cls(), self.pool_fn]

        for i, (hsize, n_blocks) in enumerate(zip(self.hidden_sizes, self.stage_sizes)):
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                layers.append(block_cls(n_hidden=hsize, strides=strides))

        layers.append(partial(jnp.mean, axis=(1, 2)))  # global average pool
        layers.append(nn.Dense(self.latent_dim))
        self.nn = nn.Sequential(layers)
        # print(self.nn)
        # print(type(self.nn))
        # self.layers = layers

    def __call__(self, x):
        return self.nn(x)
        # for i, layer in enumerate(self.layers):
        #     print(i, type(layer))
        #     x = layer(x)
        # return x


class ResNetAutoencoder(nn.Module):
    """A simple CNN autoencoder."""

    def setup(self):
        self.encoder = ResNetEncoder(
            img_shape=self.img_shape,
            latent_dim=self.latent_dim,
            nonlinearity=self.nonlinearity,
        )
        # self.decoder = ResNetDecoder(
        #     img_shape=self.img_shape,
        #     latent_dim=self.latent_dim,
        #     nonlinearity=self.nonlinearity,
        # )


ResNet18Encoder = partial(ResNetEncoder, stage_sizes=STAGE_SIZES[18],
                          stem_cls=ResNetStem, block_cls=ResNetBlock)

ResNet18Autoencoder = partial(ResNetAutoencoder, stage_sizes=STAGE_SIZES[18],
                              stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet34Autoencoder = partial(ResNetAutoencoder, stage_sizes=STAGE_SIZES[34],
                              stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet50Autoencoder = partial(ResNetAutoencoder, stage_sizes=STAGE_SIZES[50],
                              stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet101Autoencoder = partial(ResNetAutoencoder, stage_sizes=STAGE_SIZES[101],
                               stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet152Autoencoder = partial(ResNetAutoencoder, stage_sizes=STAGE_SIZES[152],
                               stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet200Autoencoder = partial(ResNetAutoencoder, stage_sizes=STAGE_SIZES[200],
                               stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)

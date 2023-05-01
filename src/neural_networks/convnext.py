from flax import linen as nn  # Linen API
import jax
import jax.numpy as jnp
from jax_models.layers import DropPath
from jax_models.models.convnext import ConvNeXtBlock, initializer
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union


__all__ = ["ConvNeXtEncoder", "ConvNeXtDecoder", "ConvNeXtAutoencoder"]


class DepthwiseConvTranpose2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: str or Sequence[Tuple[int, int]] = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.channel_multiplier * input.shape[-1],)
            )

        # conv = jax.lax.conv_general_dilated(
        #     lhs=input,
        #     rhs=w,
        #     window_strides=self.stride,
        #     padding=self.padding,
        #     lhs_dilation=(1,) * len(self.kernel_shape),
        #     rhs_dilation=(1,) * len(self.kernel_shape),
        #     dimension_numbers=("NHWC", "HWIO", "NHWC"),
        #     feature_group_count=input.shape[-1],
        # )
        conv = jax.lax.conv_transpose(
            lhs=input,
            rhs=w,
            strides=self.stride,
            padding=self.padding,
            rhs_dilation=(1,) * len(self.kernel_shape),
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv


class ConvNeXtBlockTranspose(nn.Module):
    dim: int = 256
    layer_scale_init_value: float = 1e-6
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = DepthwiseConvTranpose2D((7, 7), weights_init=initializer, name="upconv")(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer, name="pwconv2")(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x

        x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt Encoder

    Attributes:
        depths (list or tuple): Depths for every block
        dims (list or tuple): Embedding dimension for every stage.
        drop_path (float): Dropout value for DropPath. Default is 0.1
        layer_scale_init_value (float): Initialization value for scale. Default is 1e-6.
        head_init_scale (float): Initialization value for head. Default is 1.0.
        attach_head (bool): Whether to attach MLP head. Default is False.
        latent_dim (int): Dimension of latent space. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.
    """
    depths: Sequence = (3, 3, 9, 3)
    dims: Sequence = (96, 192, 384, 768)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = True
    latent_dim: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dp_rates = jnp.linspace(0, self.drop_path, sum(self.depths))
        curr = 0

        # Stem
        x = nn.Conv(
            features=self.dims[0],
            kernel_size=(4, 4),
            strides=4,
            kernel_init=initializer,
            name="downsample_layers00"
        )(inputs)
        x = nn.LayerNorm(name="downsample_layers01")(x)

        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0],
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"stages0{j}",
            )(x, deterministic)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"downsample_layers{i + 1}0")(x)

            y = x
            x = nn.Conv(
                features=self.dims[i + 1],
                kernel_size=(2, 2),
                strides=2,
                kernel_init=initializer,
                name=f"downsample_layers{i + 1}1",
            )(x)

            for j in range(self.depths[i + 1]):
                y = x
                x = ConvNeXtBlock(
                    self.dims[i + 1],
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i + 1}{j}",
                )(x, deterministic)

            curr += self.depths[i + 1]

        if self.attach_head:
            x = nn.LayerNorm(name="norm")(jnp.mean(x, [1, 2]))
            x = nn.Dense(self.latent_dim, kernel_init=initializer, name="head")(x)
        return x


class ConvNeXtDecoder(nn.Module):
    """
    ConvNeXt Decoder

    Attributes:
        depths (list or tuple): Depths for every block
        dims (list or tuple): Embedding dimension for every stage.
        drop_path (float): Dropout value for DropPath. Default is 0.1
        layer_scale_init_value (float): Initialization value for scale. Default is 1e-6.
        head_init_scale (float): Initialization value for head. Default is 1.0.
        attach_head (bool): Whether to attach MLP head. Default is False.
        headd_out_dim (int): Output dimension of head. Only works if attach_head is True.
            Needs to correspond to the flattened input dimension of the convolutional part of the network.
            Default is 384.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.
    """
    depths: Sequence = (3, 9, 3, 3)
    dims: Sequence = (384, 192, 96, 1)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = True
    downsampled_img_dim: Sequence = (2, 2, 768)
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dp_rates = jnp.linspace(0, self.drop_path, sum(self.depths))
        curr = 0

        if self.attach_head:
            x = nn.Dense(math.prod(self.downsampled_img_dim), kernel_init=initializer, name="head")(x)
            x = x.reshape((
                x.shape[0],  # batch size
                *self.downsampled_img_dim
            ))  # unflatten

        # Upsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"upsample_layers{i}0")(x)
            x = nn.ConvTranspose(
                features=self.dims[i],
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_init=initializer,
                name=f"upsample_layers{i}1",
            )(x)

            for j in range(self.depths[i]):
                x = ConvNeXtBlock(
                    self.dims[i],
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i}{j}",
                )(x, deterministic)

            curr += self.depths[i]

        # Stem
        x = nn.ConvTranspose(
            features=self.dims[3],
            kernel_size=(4, 4),
            strides=(4, 4),
            kernel_init=initializer,
            name="upsample_layers30"
        )(x)
        x = nn.LayerNorm(name="upsample_layers31")(x)

        for j in range(self.depths[3]):
            x = ConvNeXtBlock(
                self.dims[3],
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"stages3{j}",
            )(x, deterministic)
        curr += self.depths[3]

        # clip to [-1, 1]
        x = -1.0 + 2 * nn.sigmoid(x)

        return x


class ConvNeXtAutoencoder(nn.Module):
    """A simple CNN autoencoder."""
    img_shape: Tuple[int, int, int]
    latent_dim: int
    depths: Sequence = (3, 3, 9, 3)
    dims: Sequence = (96, 192, 384, 768)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = True
    deterministic: bool = True  # if false, dropout is applied

    def setup(self):
        self.encoder = ConvNeXtEncoder(
            depths=self.depths,
            dims=self.dims,
            drop_path=self.drop_path,
            layer_scale_init_value=self.layer_scale_init_value,
            head_init_scale=self.head_init_scale,
            attach_head=self.attach_head,
            latent_dim=self.latent_dim,
            deterministic=self.deterministic,
        )

        # the size of the image after the encoder, but before the head (i.e. before the MLP)
        # the first layer uses a stride of 4, the other 3 use a stride of 2
        downsampled_img_dim = (
            int(self.img_shape[0] / (4 * 2**3)),
            int(self.img_shape[1] / (4 * 2**3)),
            self.dims[-1],
        )
        print("Computed downsampled image dimension:", downsampled_img_dim)

        # compute the decoder dimensions
        decoder_dims = tuple(reversed(self.dims[:-1])) + (self.img_shape[-1], )
        print("Computed decoder dimensions:", decoder_dims)

        self.decoder = ConvNeXtDecoder(
            depths=tuple(reversed(self.depths)),
            dims=decoder_dims,
            drop_path=self.drop_path,
            layer_scale_init_value=self.layer_scale_init_value,
            head_init_scale=self.head_init_scale,
            attach_head=self.attach_head,
            downsampled_img_dim=downsampled_img_dim,
            deterministic=self.deterministic,
        )

    def __call__(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

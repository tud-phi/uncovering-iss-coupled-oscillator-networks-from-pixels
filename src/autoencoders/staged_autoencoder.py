from flax import linen as nn  # Linen API
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from jax import debug, lax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Sequence, Tuple, Union

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None,
    str,
    lax.Precision,
    Tuple[str, str],
    Tuple[lax.Precision, lax.Precision],
]

default_kernel_init = initializers.lecun_normal()


class StagedAutoencoder(nn.Module):
    backbone: nn.Module
    config_dim: int  # dimensionality of the configuration space
    mirror_head: bool = False  # weather to use the same nn parameters in the encoder and decoder head

    def setup(self):
        if self.mirror_head:
            self.head = MirroredDense(features=self.config_dim)
        else:
            self.encoder_head = nn.Dense(features=self.config_dim)
            self.decoder_head = nn.Dense(features=self.backbone.latent_dim)

    def __call__(self, x: Array, use_head: bool = True):
        x = self.encode(x, use_head=use_head)
        x = self.decode(x, use_head=use_head)
        return x

    def encode(self, x: Array, use_head: bool = True):
        x = self.backbone.encode(x)
        if use_head:
            if self.mirror_head:
                x = self.head.encode(x)
            else:
                x = self.encoder_head(x)
        return x

    def decode(self, x: Array, use_head: bool = True):
        if use_head:
            if self.mirror_head:
                x = self.head.decode(x)
            else:
                x = self.decoder_head(x)

        x = self.backbone.decode(x)
        return x


class MirroredDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        initializers.zeros_init()
    )

    def setup(self):
        assert self.features == 1, (
            "Currently, the inverse of the kernel is only implemented for 1D kernels. Otherwise, we need to make sure that the kernel is positive definite and therefore invertible."
        )

        self.kernel = self.param(
            'kernel',
            self.kernel_init,
            (self.features, self.features),
            self.param_dtype,
        )
        if self.use_bias:
            self.bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            )
        else:
            self.bias = None

    def encode(self, x: Array) -> Array:
        x, kernel, bias = promote_dtype(x, self.kernel, self.bias, dtype=self.dtype)

        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.bias is not None:
            # add bias
            x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))

        return x

    def decode(self, x: Array) -> Array:
        x, kernel, bias = promote_dtype(x, self.kernel, self.bias, dtype=self.dtype)

        if self.bias is not None:
            # subtract bias
            x -= jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))

        # compute the inverse of the kernel
        # TODO: make sure that kernel stays positive definite for multi-dimensional kernels
        inv_kernel = jnp.linalg.inv(kernel)

        x = lax.dot_general(
            x,
            inv_kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        return x

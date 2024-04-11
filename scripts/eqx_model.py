import jax
import jax.numpy as jnp
from utils_jax import *
import equinox as eqx
import optax
from typing import List, Union, Tuple, Dict, Optional, Any
from typing import Callable
from jaxtyping import Array, Float, Int, PyTree

class SimpleNet(eqx.Module):
    layers: list
    in_channels:int
    def __init__(self, in_channels, key):
        self.in_channels = in_channels
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Conv2d(in_channels, 32, kernel_size=3, padding=0, key=key1),
            eqx.nn.Conv2d(32, 16, kernel_size=3, padding=0, key=key2),
            jnp.ravel,
            eqx.nn.Linear(16*28*28, 10, key=key3),
        ]

    def __call__(self, x: Float[Array, "-1 h w"])-> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


class FlippedQuanv3x3(eqx.Module):
  weight: jax.Array
  bias: jax.Array
  stride: int
  pad: tuple|None
  pad_h: int
  pad_w: int

  def __init__(self, in_channels, out_channels, stride, padding, key):
    wkey, bkey = jax.random.split(key,2)
    self.weight = jax.random.normal(shape=[out_channels, in_channels, 15], key=wkey)
    self.bias = jax.random.normal(shape=[out_channels, 1], key=bkey)
    self.stride = stride
    self.pad = padding
    self.pad_h, self.pad_w = padding if padding is not None else (0,0)

  def __call__(self, x):
    # x has shape ( ,c_in, h, w)
    # weight has shape (c_out, c_in, 15)
    # bias has shape (c_out, 1)
    c_in, h_in, w_in = x.shape[-3], x.shape[-2], x.shape[-1]
    patches = extract_patches(x, patch_size=3, stride=self.stride, padding=self.pad)
    h_out = (h_in-3+2*self.pad_h)//self.stride +1
    w_out = (w_in-3+2*self.pad_w)//self.stride +1
    out = vmap_vmap_single_kernel_op_through_extracted_patches(self.weight, patches)
    out = out + self.bias
    return out.reshape((-1, h_out, w_out))

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jrng_key = jax.random.PRNGKey(0)
    test_img = jnp.stack([jnp.arange(32*32*1, dtype=jnp.float_).reshape((32, 32))]*3*5, axis = 0).reshape((5,3,32,32))
    print(test_img.shape)
    test_module = FlippedQuanv3x3(in_channels=3, out_channels=2, stride=1, padding=(0, 0), key=jrng_key)
    print(test_module)
    test_out = jax.vmap(test_module)(test_img)
    print(test_out.shape)

    test_simplemodel = SimpleNet(in_channels=3, key=jrng_key)
    print(test_simplemodel)
    test_out = jax.vmap(test_simplemodel)(test_img)
    print(test_out.shape)
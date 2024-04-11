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

class DataReUploadingLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    in_dim:int
    out_dim:int
    n_qubits:int
    n_reps:int

    def __init__(self, in_dim, out_dim, n_qubits, n_reps, key):
        assert 2 ** n_qubits >= out_dim
        assert 4 ** n_qubits >= in_dim
        wkey, bkey = jax.random.split(key, 2)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        param_dim = 4 ** n_qubits - 1
        self.weight = jax.random.normal(shape=[self.n_reps, param_dim], key=wkey)
        self.bias = jax.random.normal(shape=[self.out_dim], key=bkey)

    def generate_observables(self):
        observables = []
        for i in range(self.out_dim):
            temp_bitstring = '{0:b}'.format(i).zfill(self.n_qubits)
            ob = jnp.outer(bitstring_to_state(temp_bitstring), bitstring_to_state(temp_bitstring))
            observables.append(ob)
        return jnp.asarray(observables)

    def get_pauli_string_tensor_list(self):
        return generate_pauli_tensor_list(generate_nqubit_pauli_strings(self.n_qubits))

    def get_pad_size(self):
        return 4 ** self.n_qubits - self.in_dim

    def __call__(self, x):
        # x has size (batchsize, in_dim)
        # pad x
        x = jnp.pad(x, (0, self.get_pad_size()))

        out = linear_layer_func(
            padded_data=x,
            params=self.weight,
            pauli_string_tensor_list=self.get_pauli_string_tensor_list(),
            observables=self.generate_observables(),
            n_qubits=self.n_qubits
        )
        out = out + self.bias
        return out


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jrng_key = jax.random.PRNGKey(0)
    test_img = jnp.stack([jnp.arange(32*32*1, dtype=jnp.float_).reshape((32, 32))]*3*5, axis = 0).reshape((5,3,32,32))
    print(test_img.shape)
    test_conv_module = FlippedQuanv3x3(in_channels=3, out_channels=2, stride=1, padding=(0, 0), key=jrng_key)
    print(test_conv_module)
    test_out = jax.vmap(test_conv_module)(test_img)
    print(test_out.shape)

    test_linear_module = DataReUploadingLinear(in_dim=45, out_dim=6, n_qubits=3, n_reps=10, key=jrng_key)
    print(test_linear_module)
    test_data = jnp.arange(45*16, dtype=jnp.float_).reshape((16,45))
    print(test_data.shape)
    test_out_linear = jax.vmap(test_linear_module)(test_data)
    print(test_out_linear.shape)

    test_simplemodel = SimpleNet(in_channels=3, key=jrng_key)
    print(test_simplemodel)
    test_out = jax.vmap(test_simplemodel)(test_img)
    print(test_out.shape)


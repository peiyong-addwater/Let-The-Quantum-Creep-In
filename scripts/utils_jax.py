import jax
import jax.numpy as jnp
import functools, itertools

#jax.config.update("jax_enable_x64", True)

ket = {
    '0':jnp.array([1,0]),
    '1':jnp.array([0,1]),
    '+':(jnp.array([1,0]) + jnp.array([0,1]))/jnp.sqrt(2),
    '-':(jnp.array([1,0]) - jnp.array([0,1]))/jnp.sqrt(2)
}

pauli = {
    'I':jnp.array([[1,0],[0,1]]),
    'X':jnp.array([[0,1],[1,0]]),
    'Y':jnp.array([[0, -1j],[1j, 0]]),
    'Z':jnp.array([[1,0],[0,-1]])
}

def tensor_product(*args):
  input_list = [a for a in args]
  return functools.reduce(jnp.kron, input_list)

def multi_qubit_identity(n_qubits:int)->jnp.ndarray:
  assert n_qubits>0
  if n_qubits == 1:
    return pauli['I']
  else:
    return tensor_product(*[pauli['I'] for _ in range(n_qubits)])

def pauli_dict_func(key):
    return pauli[key]

def pauli_dict_func_multiple_keys(keys):
    return list(map(pauli_dict_func, keys))

def pauli_string_tensor_prod(pauli_string:str):
    paulis_char = list(pauli_string)
    paulis_mat = pauli_dict_func_multiple_keys(paulis_char)
    return tensor_product(*paulis_mat)

def generate_nqubit_pauli_strings(n_qubits:int):
    assert n_qubits>0
    pauli_labels = ['I', 'X', 'Y', 'Z']
    pauli_strings = []
    for labels in itertools.product(pauli_labels, repeat=n_qubits):
        pauli_str = "".join(labels)
        if pauli_str != 'I'*n_qubits:
            pauli_strings.append(pauli_str)
    return pauli_strings

def generate_pauli_tensor_list(pauli_strings:list):
    return list(map(pauli_string_tensor_prod, pauli_strings))

su4_generators = generate_pauli_tensor_list(
    generate_nqubit_pauli_strings(2)
)

su32_generators = generate_pauli_tensor_list(
    generate_nqubit_pauli_strings(5)
)

su8_generators = generate_pauli_tensor_list(
    generate_nqubit_pauli_strings(3)
)

su16_generators = generate_pauli_tensor_list(
    generate_nqubit_pauli_strings(4)
)

def su32_op(
    params:jnp.ndarray
):
    generator = jnp.einsum("i, ijk - >jk", params, jnp.asarray(su32_generators))
    return jax.scipy.linalg.expm(1j*generator)

def su4_op(
    params:jnp.ndarray
):
    generator = jnp.einsum("i, ijk - >jk", params, jnp.asarray(su4_generators))
    return jax.scipy.linalg.expm(1j*generator)

def su8_op(
    params:jnp.ndarray
):
    generator = jnp.einsum("i, ijk - >jk", params, jnp.asarray(su8_generators))
    return jax.scipy.linalg.expm(1j*generator)

def su16_op(
    params:jnp.ndarray
):
    generator = jnp.einsum("i, ijk - >jk", params, jnp.asarray(su16_generators))
    return jax.scipy.linalg.expm(1j*generator)

def measure_sv(
    state:jnp.ndarray,
    observable:jnp.ndarray
    ):
  """
  Measure a statevector with a Hermitian observable.
  Note: No checking Hermitianicity of the observable or whether the observable
  has all real eigenvalues or not
  """
  expectation_value = jnp.dot(jnp.conj(state.T), jnp.dot(observable, state))
  return jnp.real(expectation_value)

def measure_dm(
    rho:jnp.ndarray,
    observable:jnp.ndarray
):
  """
  Measure a density matrix with a Hermitian observable.
  Note: No checking Hermitianicity of the observable or whether the observable
  has all real eigenvalues or not.
  """
  product = jnp.dot(rho, observable)

  # Calculate the trace, which is the sum of diagonal elements
  trace = jnp.trace(product)

  # The expectation value should be real for physical observables
  return jnp.real(trace)

# assuming the input patch (hermitianized) has shape (c, h, w)
# assuming the input set statevectors has shape (c, 2**n)
# assuming we have a list of (state, observable) pairs
vmap_measure_sv_ob_pairs = jax.vmap(lambda pair: measure_sv(pair[0], pair[1]), in_axes=0, out_axes=0)
# assuming the input set desnity matrices has shape (c, 2**n, 2**n)
# assuming we have a list of (rho, observable) pairs
vmap_measure_dm_ob_pairs = jax.vmap(lambda pair: measure_dm(pair[0], pair[1]), in_axes=0, out_axes=0)

# vmap through different observables
vmap_measure_sv = jax.vmap(measure_sv, in_axes=(None, 0), out_axes=0)
vmap_measure_dm = jax.vmap(measure_dm, in_axes=(None, 0), out_axes=0)

def bitstring_to_state(bitstring:str):
  """
  Convert a bit string, like '0101001' or '+-+-101'
  to a statevector. Each character in the bitstring must be among
  0, 1, + and -
  """
  assert len(bitstring)>0
  for c in bitstring:
    assert c in ['0', '1', '+', '-']
  single_qubit_states = [ket[c] for c in bitstring]
  return tensor_product(*single_qubit_states)


# utilities for the flipped quanvolution kernel
def extract_patches(image, patch_size, stride, padding=None):
    """
    Extracts patches from an image with multiple input channels and optional custom padding.

    Args:
        image (jnp.ndarray): Input image tensor of shape (in_channels, height, width).
        patch_size (int): Size of the square patches to extract.
        stride (int): Stride between patches.
        padding (tuple): Padding value(s) for each dimension.

    Returns:
        jnp.ndarray: Tensor of extracted patches of shape (num_patches, in_channels, patch_size, patch_size).
    """

    in_channels, height, width = image.shape

    pad_h, pad_w = padding if padding is not None else (0, 0)


    image = jnp.pad(image, [(0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='constant') if padding is not None else image


    _, height, width = image.shape


    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1

    patch_indices = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]

    patches = jnp.stack([image[:, i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]
                         for i, j in patch_indices])

    return patches


def generate_2q_param_state(theta):
  state = bitstring_to_state('00')
  state = jnp.dot(
      su4_op(theta),
      state
  )
  return state

vmap_generate_2q_param_state = jax.vmap(generate_2q_param_state, in_axes=0, out_axes = 0)

def single_kernel_op(thetas, patch):
  # patch has shape (c_in, h, w)
  # thetas has shape (c_in, 4^2-1) for SU4 gates
  n_theta = thetas.shape[0]
  n_channel = patch.shape[0]
  assert n_theta == n_channel, "Thetas and patch must have the same number of channels."
  states = vmap_generate_2q_param_state(thetas)
  patch = jnp.pad(patch, [(0,0),(0,1),(0,1)], mode='constant')
  herm_patch = (jnp.einsum("ijk->ikj", patch)+patch)/2
  channel_out = vmap_measure_sv_ob_pairs([states, herm_patch])
  return jnp.sum(channel_out, axis = 0)/n_theta

vmap_single_kernel_op_through_extracted_patches = jax.vmap(single_kernel_op, in_axes=(None, 0), out_axes=0)

# For multiple channel output
# parameter has shape (c_out, c_in, 4**2-1) for SU4 gates
vmap_vmap_single_kernel_op_through_extracted_patches = jax.vmap(vmap_single_kernel_op_through_extracted_patches, in_axes=(0, None), out_axes=0)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    seed = 0

    jrng_key = jax.random.PRNGKey(seed)

    test_params = jax.random.normal(shape=[4 ** 2 - 1], key=jrng_key)

    print(
        jnp.einsum(
            "ij,jk->ik",
            jnp.transpose(jnp.conjugate(su4_op(test_params))),
            su4_op(test_params)
        )
    )

    print(
        jnp.einsum(
            "ij,jk->ik",
            su4_op(test_params),
            jnp.transpose(jnp.conjugate(su4_op(test_params)))
        )
    )

    test_patch = jax.random.normal(shape=[3, 4, 4], key=jrng_key)
    test_herm_patch = (jnp.einsum("ijk->ikj", test_patch) + test_patch) / 2
    print(test_herm_patch)
    print()
    test_sv = jnp.stack([bitstring_to_state('++')] * 3, axis=0)
    print(test_sv)
    print()
    print(vmap_measure_sv_ob_pairs([test_sv, test_herm_patch]))
    print()
    for sv, ob in zip(test_sv, test_herm_patch):
        print(measure_sv(sv, ob))

    test_img = jnp.stack([jnp.arange(32 * 32 * 1).reshape((32, 32)), jnp.arange(32 * 32 * 1).reshape((32, 32)),
                          jnp.arange(32 * 32 * 1).reshape((32, 32))], axis=0)

    print(test_img[:, :4, :4])

    patches = extract_patches(test_img, patch_size=4, stride=2, padding=(0, 0))
    print(patches.shape)
    print(patches[0].shape)
    print(patches[0])

    test_params = jax.random.normal(shape=[3, 4 ** 2 - 1], key=jrng_key)
    test_img = jnp.stack([jnp.arange(32*32*1).reshape((32, 32))]*3, axis = 0)
    print(test_img.shape)
    test_patches = extract_patches(test_img, patch_size=3, stride=1, padding=(1, 1))
    print(test_patches.shape)
    print(test_params.shape)
    print(vmap_generate_2q_param_state(test_params))
    test_out = vmap_single_kernel_op_through_extracted_patches(test_params, test_patches)
    print(test_out.shape)
    h = (32 - 3 + 1 * 2) // 1 + 1
    print(h ** 2)

    test_params2 = jax.random.normal(shape=[4, 3, 4 ** 2 - 1], key=jrng_key)
    test_out2 = vmap_vmap_single_kernel_op_through_extracted_patches(test_params2, test_patches)
    print(test_out2.shape)
    test_out_2_features = test_out2.reshape((-1, 32, 32))
    print(test_out_2_features.shape)
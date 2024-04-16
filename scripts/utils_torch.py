import torch
import functools, itertools

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cfloat won't pass the unitary check
# but it saves a lot of memory
# and significantly speeds up the training
# torch.cdouble will pass the unitary check
# but it is very slow
COMPLEX_DTYPE = torch.cfloat #torch.cdouble
REAL_DTYPE = torch.float

ket = {
    '0':torch.tensor([1.,0.], dtype = COMPLEX_DTYPE, device=DEVICE),
    '1':torch.tensor([0.,1.], dtype = COMPLEX_DTYPE, device=DEVICE),
    '+': (torch.tensor([1,0], dtype = COMPLEX_DTYPE, device=DEVICE) + torch.tensor([0, 1], dtype = COMPLEX_DTYPE, device=DEVICE)) / torch.sqrt(torch.tensor(2, dtype = COMPLEX_DTYPE, device=DEVICE)),
    '-': (torch.tensor([1,0], dtype = COMPLEX_DTYPE, device=DEVICE) - torch.tensor([0, 1], dtype = COMPLEX_DTYPE, device=DEVICE)) / torch.sqrt(torch.tensor(2, dtype = COMPLEX_DTYPE, device=DEVICE))
}

pauli = {
    'I':torch.tensor([[1.,0.],[0.,1.]], dtype = COMPLEX_DTYPE, device=DEVICE),
    'X':torch.tensor([[0.,1.],[1.,0.]], dtype = COMPLEX_DTYPE, device=DEVICE),
    'Y':torch.tensor([[0., -1.j],[1.j, 0]], dtype = COMPLEX_DTYPE, device=DEVICE),
    'Z':torch.tensor([[1.,0.],[0.,-1.]], dtype = COMPLEX_DTYPE, device=DEVICE)
}

def tensor_product(*args):
  input_list = [a for a in args]
  return functools.reduce(torch.kron, input_list)

def multi_qubit_identity(n_qubits:int):
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
        params
):
    genreator = torch.einsum("i,ijk->jk", params, torch.stack(su32_generators))
    return torch.matrix_exp(1j*genreator)

def su16_op(
        params
):
    genreator = torch.einsum("i,ijk->jk", params, torch.stack(su16_generators))
    return torch.matrix_exp(1j*genreator)

def su8_op(
        params
):
    genreator = torch.einsum("i,ijk->jk", params, torch.stack(su8_generators))
    return torch.matrix_exp(1j*genreator)

def su4_op(
        params
):
    genreator = torch.einsum("i,ijk->jk", params, torch.stack(su4_generators))
    return torch.matrix_exp(1j*genreator)

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

def measure_sv(
    state,
    observable
    ):
  """
  Measure a statevector with a Hermitian observable.
  Note: No checking Hermitianicity of the observable or whether the observable
  has all real eigenvalues or not
  """
  expectation_value = torch.conj(state)@observable@state
  return torch.real(expectation_value)

def measure_dm(
    rho,
    observable
):
  """
  Measure a density matrix with a Hermitian observable.
  Note: No checking Hermitianicity of the observable or whether the observable
  has all real eigenvalues or not.
  """
  product = torch.matmul(rho, observable)

  # Calculate the trace, which is the sum of diagonal elements
  trace = torch.trace(product)

  # The expectation value should be real for physical observables
  return torch.real(trace)

vmap_measure_sv = torch.vmap(measure_sv, in_dims=(None, 0), out_dims=0)
vmap_measure_dm = torch.vmap(measure_dm, in_dims=(None, 0), out_dims=0)

# assuming the input patch observables (hermitianized) has shape (batchsize,n_patches, c, h, w)
# assuming the input set statevectors has shape (c, 2**n)
# output should have the shape (batchsize,n_patches, channel)
vmap_measure_channel_sv_batched_ob = torch.vmap(measure_sv, in_dims = (-2, -3),out_dims=-1)


# assuming the input set desnity matrices has shape (batchsize, n_patches, c, 2**n, 2**n)
# output should have the shape (batchsize,n_patches, channel)
vmap_measure_channel_dm_batched_ob = torch.vmap(measure_dm, in_dims = (-2, -3),out_dims=-1)


# utility functions for the flipped quanvolution
def extract_patches(image, patch_size, stride, padding=None):
    """
    Extracts patches from an image with multiple input channels and optional custom padding.

    Args:
        image (torch.Tensor): Input image tensor of shape (in_channels, height, width).
        patch_size (int): Size of the square patches to extract.
        stride (int): Stride between patches.
        padding (tuple): Padding value(s) for each dimension.

    Returns:
        torch.Tensor: Tensor of extracted patches of shape (num_patches, in_channels, patch_size, patch_size).
    """
    in_channels, height, width = image.shape[-3], image.shape[-2], image.shape[-1]
    pad_l, pad_r, pad_t, pad_b = padding if padding is not None else (0,0,0,0)

    if padding is not None:
        image = torch.nn.functional.pad(image, (pad_l, pad_r, pad_t, pad_b), mode='constant')
    else:
        image = image

    height, width = image.shape[-2],  image.shape[-1]
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[..., i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]
            patches.append(patch)

    patches = torch.stack(patches, dim=-4)
    return patches # has shape (batchsize, n_patches, channel, h, w)

def generate_2q_param_state(theta):
  state = bitstring_to_state('00')
  state = torch.matmul(su4_op(theta), state)
  return state

vmap_generate_2q_param_state = torch.vmap(generate_2q_param_state, in_dims=0, out_dims = 0)

def single_kernel_op_over_batched_patches(thetas, patch):
  # patch has shape (c_in, h, w)
  # thetas has shape (c_in, 4^2-1) for SU4 gates
  n_theta = thetas.shape[-2]
  n_channel = patch.shape[-3]
  assert n_theta == n_channel, "Thetas and patch must have the same number of channels."
  states = vmap_generate_2q_param_state(thetas)
  #print("States shape", states.shape)
  patch = torch.nn.functional.pad(patch, (0, 1, 0, 1), mode='constant')
  patch_t = torch.einsum("...jk->...kj", patch)
  herm_patch = (patch_t+patch)/2 # has dim (batchsize, num_patches, c, h, w)
  #print("Herm patch shape", herm_patch.shape)
  channel_out = vmap_measure_channel_sv_batched_ob(states, herm_patch) # has dim (batchsize,n_patches, c)
  return torch.sum(channel_out, axis = -1)/n_theta # has dim (batchszie, n_patches)

# For multiple channel output
# parameter has shape (c_out, c_in, 4**2-1) for SU4 gates
vmap_vmap_single_kernel_op_through_extracted_patches = torch.vmap(single_kernel_op_over_batched_patches, in_dims=(0, None), out_dims=-2) # output has dim (batchsize, c_out, n_patches)

# Utility functions for the data reuploading linear layer
def data_encode_unitary(padded_data, t):
  original_dim = padded_data.shape[-1]
  new_dim = torch.sqrt(torch.tensor(original_dim)).type(torch.int)
  data = torch.reshape(padded_data, (new_dim, new_dim))
  generator = (data + torch.einsum("...jk->...kj", data))/2
  return torch.linalg.matrix_exp(1.0j*generator*t)

def su_n(params, pauli_string_tensor_list):
  # params has dim 4**n-1
  paulis = torch.stack(pauli_string_tensor_list)
  generator = torch.einsum("i,ijk->jk", params, paulis)
  return torch.linalg.matrix_exp(1.0j*generator)

def linear_layer_func(padded_data, params, pauli_string_tensor_list, observables, n_qubits):
  n_rep = params.shape[0]
  state = bitstring_to_state("+"*n_qubits)
  for i in range(n_rep):
    data_unitary = data_encode_unitary(padded_data, 1/n_rep)
    #print(data_unitary.shape)
    #print(state.shape)
    state = torch.matmul(
      data_unitary,
      state
    )
    sun_gate = su_n(params[i], pauli_string_tensor_list)
    #print(sun_gate.shape)
    state = torch.matmul(
      sun_gate,
      state
    )
  return vmap_measure_sv(state, observables)

vmap_batch_linear_layer_func = torch.vmap(linear_layer_func, in_dims=(0, None, None, None, None), out_dims=0)



if __name__ == '__main__':
    test_params = torch.randn((3, 4 ** 2 - 1), device=DEVICE).type(COMPLEX_DTYPE)
    #test_data = torch.randn(4 ** 2, device=device).type(COMPLEX_DTYPE)
    test_obs = torch.stack([torch.outer(bitstring_to_state('00'), bitstring_to_state('00')),
                            torch.outer(bitstring_to_state('01'), bitstring_to_state('01')),
                            torch.outer(bitstring_to_state('10'), bitstring_to_state('10')),
                            torch.outer(bitstring_to_state('11'), bitstring_to_state('11'))])
    test_pauli_string_tensor_list = generate_pauli_tensor_list(generate_nqubit_pauli_strings(2))

    #test_out = linear_layer_func(test_data, test_params, test_pauli_string_tensor_list, test_obs, 2)
    test_data = torch.randn((3, 4 ** 2), device=DEVICE).type(COMPLEX_DTYPE)
    test_out = vmap_batch_linear_layer_func(test_data, test_params, test_pauli_string_tensor_list, test_obs, 2)
    print(test_out)
    print(torch.sum(test_out))
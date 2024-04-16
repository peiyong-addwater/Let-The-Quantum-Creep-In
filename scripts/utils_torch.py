import torch
import functools, itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMPLEX_DTYPE = torch.cfloat #torch.cdouble
REAL_DTYPE = torch.float

ket = {
    '0':torch.tensor([1.,0.], dtype = COMPLEX_DTYPE, device=device),
    '1':torch.tensor([0.,1.], dtype = COMPLEX_DTYPE, device=device),
    '+':(torch.tensor([1,0], dtype = COMPLEX_DTYPE, device=device) + torch.tensor([0,1], dtype = COMPLEX_DTYPE, device=device))/torch.sqrt(torch.tensor(2, dtype = COMPLEX_DTYPE, device=device)),
    '-':(torch.tensor([1,0], dtype = COMPLEX_DTYPE, device=device) - torch.tensor([0,1], dtype = COMPLEX_DTYPE, device=device))/torch.sqrt(torch.tensor(2, dtype = COMPLEX_DTYPE, device=device))
}

pauli = {
    'I':torch.tensor([[1.,0.],[0.,1.]], dtype = COMPLEX_DTYPE, device=device),
    'X':torch.tensor([[0.,1.],[1.,0.]], dtype = COMPLEX_DTYPE, device=device),
    'Y':torch.tensor([[0., -1.j],[1.j, 0]], dtype = COMPLEX_DTYPE, device=device),
    'Z':torch.tensor([[1.,0.],[0.,-1.]], dtype = COMPLEX_DTYPE, device=device)
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



if __name__ == '__main__':
    test_su32_params = torch.randn(4**5-1, device=device).type(COMPLEX_DTYPE)
    print(
        su32_op(test_su32_params)
    )
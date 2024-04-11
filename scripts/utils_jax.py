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



if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    seed = 0

    test_params = jax.random.normal(shape=[4 ** 2 - 1], key=jax.random.PRNGKey(seed))

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
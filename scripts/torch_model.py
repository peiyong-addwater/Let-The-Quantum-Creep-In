import torch
from torch import nn as nn
import torch.nn.functional as F
from utils_torch import *

class SimpleNet(nn.Module):
    def __init__(self, in_channels):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.Flatten(),
            nn.Linear(16*28*28, 10)
        )

    def forward(self, x):
        return self.layers(x)

class FlippedQuanv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(FlippedQuanv3x3, self).__init__()
        super(FlippedQuanv3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.pad_l, self.pad_r, self.pad_t, self.pad_b = padding if padding is not None else (0, 0, 0, 0)
        self.pad = (self.pad_l, self.pad_r, self.pad_t, self.pad_b)
        self.weight = torch.nn.Parameter(torch.randn((out_channels, in_channels, 4 ** 2 - 1)).type(COMPLEX_DTYPE))
        self.bias = torch.nn.Parameter(torch.randn((out_channels, 1)).type(COMPLEX_DTYPE))

    def forward(self, x):
        # x has shape (batchsize ,c_in, h, w)
        # weight has shape (c_out, c_in, 15)
        # bias has shape (c_out, 1)
        x = x.type(COMPLEX_DTYPE)
        c_in, h_in, w_in = x.shape[-3], x.shape[-2], x.shape[-1]
        patches = extract_patches(x, patch_size=3, stride=self.stride, padding=self.pad)
        h_out = (h_in - 3 + (self.pad_t + self.pad_b)) // self.stride + 1
        w_out = (w_in - 3 + (self.pad_l + self.pad_r)) // self.stride + 1

        out = vmap_vmap_single_kernel_op_through_extracted_patches(self.weight, patches)

        out = out + self.bias
        return out.reshape((-1, self.out_channels, h_out, w_out)).type(REAL_DTYPE)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}, padding={self.padding}'

class DataReUploadingLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_qubits, n_reps):
        super(DataReUploadingLinear, self).__init__()
        assert 2 ** n_qubits >= out_dim
        assert 4 ** n_qubits >= in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.pauli_string_tensor_list = generate_pauli_tensor_list(generate_nqubit_pauli_strings(n_qubits))
        self.param_dim = 4 ** n_qubits - 1
        self.params = torch.nn.Parameter(torch.randn((self.n_reps, self.param_dim)).type(COMPLEX_DTYPE))
        self.bias = torch.nn.Parameter(torch.randn(out_dim).type(REAL_DTYPE))
        self.observables = self.generate_observables()
        self.pad_size = 4 ** n_qubits - self.in_dim

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, n_qubits={self.n_qubits}, n_reps={self.n_reps}'

    def generate_observables(self):
        observables = []
        for i in range(self.out_dim):
            temp_bitstring = '{0:b}'.format(i).zfill(self.n_qubits)
            ob = torch.outer(bitstring_to_state(temp_bitstring), bitstring_to_state(temp_bitstring))
            observables.append(ob)
        return torch.stack(observables)

    def forward(self, x):
        # x has size (batchsize, in_dim)
        # pad x
        x = torch.nn.functional.pad(x, (0, self.pad_size)).type(COMPLEX_DTYPE)
        out = vmap_batch_linear_layer_func(x, self.params, self.pauli_string_tensor_list, self.observables,
                                           self.n_qubits)
        out = out.type(REAL_DTYPE) + self.bias
        return out

class HybridNet(nn.Module):
    def __init__(self, in_channels, replacement_lvl):
        super(HybridNet, self).__init__()
        self.in_channels = in_channels
        self.replacement_lvl = replacement_lvl
        if self.replacement_lvl == 0:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3),
                nn.Conv2d(32, 16, kernel_size=3),
                nn.Flatten(),
                nn.Linear(16 * 28 * 28, 10)
            )
        elif self.replacement_lvl == 1:
            self.layers = nn.Sequential(
                FlippedQuanv3x3(in_channels=1, out_channels=32, stride=1, padding=None), # 32->30
                FlippedQuanv3x3(in_channels=32, out_channels=16, stride=1, padding=None),  # 30->28
                torch.nn.Flatten(),
                nn.Linear(16 * 28 * 28, 10)
            )
        elif self.replacement_lvl == 2:
            self.layers = torch.nn.Sequential(
                FlippedQuanv3x3(in_channels=1, out_channels=32, stride=1, padding=None),  # 32->30
                FlippedQuanv3x3(in_channels=32, out_channels=16, stride=1, padding=None),  # 30->28
                torch.nn.Flatten(),
                DataReUploadingLinear(16 * 28 * 28, 10, 7, 10)
            )
        else:
            raise ValueError("replacement_lvl should be 0, 1, or 2")

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    import torchvision
    """
    test_linear_module = DataReUploadingLinear(in_dim=45, out_dim=6, n_qubits=3, n_reps=10).to(DEVICE)
    test_data = torch.randn((16, 45), device=DEVICE).type(COMPLEX_DTYPE)
    print(test_data.shape)
    test_out = test_linear_module(test_data.to(DEVICE))
    print(test_out.shape)
    print(test_linear_module)
    """
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Pad(2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.Lambda(lambda x: x.type(COMPLEX_DTYPE))
    ])

    train_dataset = torchvision.datasets.MNIST(
        "MNIST",
        train=True,
        download=True,
        transform=preprocess,
    )
    test_dataset = torchvision.datasets.MNIST(
        "MNIST",
        train=False,
        download=True,
        transform=preprocess,
    )
    dummy_trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    dummy_testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True
    )

    dummy_x, dummy_y = next(iter(dummy_trainloader))

    print(dummy_x.shape)  # 64x32x32
    print(dummy_y.shape)  # 64
    print(dummy_y)
    print(dummy_x[0, 0, 16])

    for rpl_lvl in [0,1,2]:
        # full replacement need at least A100-40GB
        test_hybridmodel = HybridNet(in_channels=1, replacement_lvl=rpl_lvl).to(DEVICE)
        test_hybridmodel = torch.compile(test_hybridmodel)
        print(test_hybridmodel)
        if rpl_lvl == 0:
            out = test_hybridmodel(dummy_x.type(REAL_DTYPE).to(DEVICE))
        else:
            out = test_hybridmodel(dummy_x.to(DEVICE))
        print(out.shape)


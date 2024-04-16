import yaml
import torch
from torch import nn as nn
import torchmetrics
import time
import numpy as np
import torchvision
from utils_torch import COMPLEX_DTYPE

#https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
def merge_dictionaries_recursively(dict1, dict2):
  if dict2 is None: return

  for k, v in dict2.items():
    if k not in dict1:
      dict1[k] = dict()
    if isinstance(v, dict):
      merge_dictionaries_recursively(dict1[k], v)
    else:
      dict1[k] = v
class Config(object):
    # https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    def __init__(self, config_path = 'config.yaml', default_path='config_torch_default.yaml'):
        with open(config_path) as cf_file:
            cfg = yaml.safe_load(cf_file.read())
        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = yaml.safe_load(def_cf_file.read())

            merge_dictionaries_recursively(default_cfg, cfg)

        self._data = cfg

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

def train_torch(
        model,
        train_dataset,
        test_dataset,
        optim = torch.optim.SGD,
        criterion = nn.CrossEntropyLoss,
        accuracy = torchmetrics.Accuracy,
        steps = 100,
        print_every_percent = 0.1,
        batchsize = 100,
        lr = 0.001,
        device=torch.device("cpu")
):
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=True
    )

    n_train_batches = len(trainloader)
    n_test_batches = len(testloader)
    print_every_train_batch = int(n_train_batches * print_every_percent)
    print_every_test_batch = int(n_test_batches * print_every_percent)

    print(f"Number of train batches = {n_train_batches}, Number of test batches = {n_test_batches}")
    print(f"Print every train batch = {print_every_train_batch}, Print every test batch = {print_every_test_batch}")

    model.to(device)
    optimizer = optim(model.parameters(), lr=lr, momentum=0.9)
    loss = criterion()
    acc_func = accuracy(task="multiclass", num_classes=10).to(device)
    step_train_losses = []
    step_test_losses = []
    step_train_accs = []
    step_test_accs = []
    for i in range(steps):
        step_start = time.time()
        batch_train_loss = []
        batch_train_acc = []
        batch_test_loss = []
        batch_test_acc = []
        # train
        model.train()
        for batchid, (images, labels) in enumerate(trainloader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = loss(outputs, labels)
            train_loss.backward()
            optimizer.step()
            train_acc = acc_func(outputs, labels)
            batch_train_loss.append(train_loss.item())
            batch_train_acc.append(train_acc.item())
            batch_finish = time.time()

            if (batchid) % print_every_train_batch == 0:
                print(
                    f"Training at step={i}, batch={batchid}, train loss = {train_loss.item()}, train acc = {train_acc.item()}, time = {batch_finish - batch_start}")

        # eval
        model.eval()
        with torch.no_grad():
            for batchid, (images, labels) in enumerate(testloader):
                batch_start = time.time()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss = loss(outputs, labels)
                test_acc = acc_func(outputs, labels)
                batch_test_loss.append(test_loss.item())
                batch_test_acc.append(test_acc.item())
                batch_finish = time.time()
                if (batchid) % print_every_test_batch == 0:
                    print(
                        f"Testing at step={i}, batch={batchid}, test loss = {test_loss.item()}, test acc = {test_acc.item()}, time = {batch_finish - batch_start}")

        step_train_losses.append(np.mean(batch_train_loss))
        step_test_losses.append(np.mean(batch_test_loss))
        step_train_accs.append(np.mean(batch_train_acc))
        step_test_accs.append(np.mean(batch_test_acc))
        step_finish = time.time()
        print(
            f"Step {i} finished in {step_finish - step_start}, Train loss = {step_train_losses[-1]}, Test loss = {step_test_losses[-1]}; Train Acc = {step_train_accs[-1]}, Test Acc = {step_test_accs[-1]}")

    return step_train_losses, step_test_losses, step_train_accs, step_test_accs

preprocess_28_quant = torchvision.transforms.Compose([
    torchvision.transforms.Pad(2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    torchvision.transforms.Lambda(lambda x: x.type(COMPLEX_DTYPE))
])

preprocess_32_quant = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    torchvision.transforms.Lambda(lambda x: x.type(COMPLEX_DTYPE))
])

preprocess_28_classical = torchvision.transforms.Compose([
    torchvision.transforms.Pad(2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    #torchvision.transforms.Lambda(lambda x: x.type(COMPLEX_DTYPE))
])

preprocess_32_classical = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    #torchvision.transforms.Lambda(lambda x: x.type(COMPLEX_DTYPE))
])

DATASETS_Quant = {
    'MINST': {
        'train': torchvision.datasets.MNIST(
            "MNIST",
            train=True,
            download=True,
            transform=preprocess_28_quant,
        ),
        'test': torchvision.datasets.MNIST(
            "MNIST",
            train=False,
            download=True,
            transform=preprocess_28_quant,
        )
    },
    'FashionMNIST': {
        'train': torchvision.datasets.FashionMNIST(
            "FashionMNIST",
            train=True,
            download=True,
            transform=preprocess_28_quant,
        ),
        'test': torchvision.datasets.FashionMNIST(
            "FashionMNIST",
            train=False,
            download=True,
            transform=preprocess_28_quant,
        )
    },
    'CIFAR10': {
        'train': torchvision.datasets.CIFAR10(
            "CIFAR10",
            train=True,
            download=True,
            transform=preprocess_32_quant,
        ),
        'test': torchvision.datasets.CIFAR10(
            "CIFAR10",
            train=False,
            download=True,
            transform=preprocess_32_quant,
        )
    }
}

DATASETS_classical = {
    'MINST': {
        'train': torchvision.datasets.MNIST(
            "MNIST",
            train=True,
            download=True,
            transform=preprocess_28_classical,
        ),
        'test': torchvision.datasets.MNIST(
            "MNIST",
            train=False,
            download=True,
            transform=preprocess_28_classical,
        )
    },
    'FashionMNIST': {
        'train': torchvision.datasets.FashionMNIST(
            "FashionMNIST",
            train=True,
            download=True,
            transform=preprocess_28_classical,
        ),
        'test': torchvision.datasets.FashionMNIST(
            "FashionMNIST",
            train=False,
            download=True,
            transform=preprocess_28_classical,
        )
    },
    'CIFAR10': {
        'train': torchvision.datasets.CIFAR10(
            "CIFAR10",
            train=True,
            download=True,
            transform=preprocess_32_classical,
        ),
        'test': torchvision.datasets.CIFAR10(
            "CIFAR10",
            train=False,
            download=True,
            transform=preprocess_32_classical,
        )
    }
}

if __name__ == '__main__':
    from datetime import datetime
    from torch_model import HybridNet
    import json
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3, 2, 1, 0" # for 4-GPU, https://github.com/pytorch/pytorch/issues/113245#issuecomment-1909409587, not working here

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(curr_time)

    print(os.getcwd())


    config = Config(config_path='config.yaml').get('experiment')
    assert config['framework'] == 'torch'
    BATCH_SIZE = int(config['batch_size'])
    STEPS = int(config['steps'])
    LEARNING_RATE = float(config['learning_rate'])
    PRINT_EVERY_PERCENT = float(config['print_every_percent'])

    N_REPS = int(config['reps'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    print("Training Config:")
    print(config)


    for replacement_lvl in [1,2]:
        for dataset_name in ['MINST', 'FashionMNIST', 'CIFAR10']:
            if dataset_name == 'CIFAR10':
                in_channels = 3
            else:
                in_channels = 1
            if replacement_lvl == 0:
                train_dataset = DATASETS_classical[dataset_name]['train']
                test_dataset = DATASETS_classical[dataset_name]['test']
            else:
                train_dataset = DATASETS_Quant[dataset_name]['train']
                test_dataset = DATASETS_Quant[dataset_name]['test']

            # repeat training for different random seeds
            for rep in range(N_REPS):
                print(f"____Training with replacement_lvl = {replacement_lvl}____")
                print(f"++++Training on {dataset_name}++++")
                print(f"----Training with rep = {rep}----")
                model = HybridNet(in_channels, replacement_lvl)
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

                train_losses, test_losses, train_accs, test_accs = train_torch(
                    model,
                    train_dataset,
                    test_dataset,
                    steps=STEPS,
                    print_every_percent=PRINT_EVERY_PERCENT,
                    batchsize=BATCH_SIZE,
                    lr=LEARNING_RATE,
                    device=device
                )

                results = {
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accs': train_accs,
                    'test_accs': test_accs
                }

                with open(os.path.join('results', f"{curr_time}_{dataset_name}_replacement_lvl_{replacement_lvl}_rep_{rep}.json"), 'w') as f:
                    json.dump(results, f, indent=4)



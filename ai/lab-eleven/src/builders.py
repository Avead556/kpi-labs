from collections import OrderedDict

import torch
from torch import nn

DATASETS = {
    "mnist": {"channels": 1, "size": 28, "classes": 10},
    "cifar": {"channels": 3, "size": 32, "classes": 10},
}


def activation(name):
    choices = nn.ModuleDict([
        ["relu", nn.ReLU()],
        ["lrelu", nn.LeakyReLU()],
        ["tanh", nn.Tanh()],
        ["sigmoid", nn.Sigmoid()],
    ])
    return choices[name]


def conv_block(in_f, out_f, activation_name="relu", convs=1, padding=1):
    layers = [nn.Conv2d(in_f, out_f, kernel_size=3, padding=padding)]
    layers += [nn.Conv2d(out_f, out_f, kernel_size=3, padding=padding)
               for _ in range(convs - 1)]
    layers += [activation(activation_name),
               nn.BatchNorm2d(out_f),
               nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def dec_block(in_f, out_f, activation_name="relu", batch_norm=False):
    layers = [nn.Linear(in_f, out_f), activation(activation_name)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_f))
    return nn.Sequential(*layers)


class Encoder(nn.Module):

    def __init__(self, sizes, activation_name="relu", convs=1, paddings=None,
                 input_norm=False):
        super().__init__()
        paddings = paddings or [1] * (len(sizes) - 1)
        self.input_norm = nn.BatchNorm2d(sizes[0]) if input_norm else nn.Identity()
        self.blocks = nn.ModuleList([
            conv_block(in_f, out_f, activation_name, convs, padding)
            for (in_f, out_f), padding in zip(zip(sizes, sizes[1:]), paddings)
        ])

    def forward(self, x, trace=None):
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
            if trace is not None:
                trace.append(x)
        return x


class Decoder(nn.Module):

    def __init__(self, sizes, n_classes, activation_name="relu", batch_norm=False):
        super().__init__()
        self.blocks = nn.Sequential(*[
            dec_block(in_f, out_f, activation_name, batch_norm)
            for in_f, out_f in zip(sizes, sizes[1:])
        ])
        self.last = nn.Linear(sizes[-1], n_classes)

    def forward(self, x):
        return self.last(self.blocks(x))


class ModularCNN(nn.Module):

    def __init__(self, in_c, enc_sizes, dec_sizes, n_classes, image_size,
                 activation_name="relu", dec_activation=None, convs=1,
                 paddings=None, input_norm=False, dec_batch_norm=False):
        super().__init__()
        self.encoder = Encoder([in_c, *enc_sizes], activation_name, convs,
                               paddings, input_norm)
        self.encoder.eval()
        with torch.no_grad():
            flat = self.encoder(torch.zeros(1, in_c, image_size, image_size)).numel()
        self.encoder.train()
        self.decoder = Decoder([flat, *dec_sizes], n_classes,
                               dec_activation or activation_name, dec_batch_norm)

    def forward(self, x, trace=None):
        return self.decoder(self.encoder(x, trace).flatten(1))


class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(6, 6, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(6, 16, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=0)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.bn1(self.act1(self.conv1_2(self.conv1_1(x)))))
        x = self.pool2(self.bn2(self.act2(self.conv2_2(self.conv2_1(x)))))
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)


class CifarNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.act4 = nn.Tanh()
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.act5 = nn.Tanh()
        self.bn5 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.bn0(x)
        x = self.pool1(self.bn1(self.act1(self.conv1(x))))
        x = self.pool2(self.bn2(self.act2(self.conv2(x))))
        x = self.pool3(self.bn3(self.act3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.bn4(self.act4(self.fc1(x)))
        x = self.bn5(self.act5(self.fc2(x)))
        return self.fc3(x)


def lenet5_sequential():
    return nn.Sequential(OrderedDict([
        ("conv1_1", nn.Conv2d(1, 6, kernel_size=3, padding=1)),
        ("conv1_2", nn.Conv2d(6, 6, kernel_size=3, padding=1)),
        ("act1", nn.ReLU()),
        ("bn1", nn.BatchNorm2d(6)),
        ("pool1", nn.MaxPool2d(2, 2)),
        ("conv2_1", nn.Conv2d(6, 16, kernel_size=3)),
        ("conv2_2", nn.Conv2d(16, 16, kernel_size=3)),
        ("act2", nn.ReLU()),
        ("bn2", nn.BatchNorm2d(16)),
        ("pool2", nn.MaxPool2d(2, 2)),
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(5 * 5 * 16, 120)),
        ("act3", nn.ReLU()),
        ("fc2", nn.Linear(120, 84)),
        ("act4", nn.ReLU()),
        ("fc3", nn.Linear(84, 10)),
    ]))


def cifarnet_sequential():
    return nn.Sequential(OrderedDict([
        ("bn0", nn.BatchNorm2d(3)),
        ("conv1", nn.Conv2d(3, 16, 3, padding=1)), ("act1", nn.ReLU()),
        ("bn1", nn.BatchNorm2d(16)), ("pool1", nn.MaxPool2d(2, 2)),
        ("conv2", nn.Conv2d(16, 32, 3, padding=1)), ("act2", nn.ReLU()),
        ("bn2", nn.BatchNorm2d(32)), ("pool2", nn.MaxPool2d(2, 2)),
        ("conv3", nn.Conv2d(32, 64, 3, padding=1)), ("act3", nn.ReLU()),
        ("bn3", nn.BatchNorm2d(64)), ("pool3", nn.MaxPool2d(2, 2)),
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(4 * 4 * 64, 256)), ("act4", nn.Tanh()),
        ("bn4", nn.BatchNorm1d(256)),
        ("fc2", nn.Linear(256, 64)), ("act5", nn.Tanh()),
        ("bn5", nn.BatchNorm1d(64)),
        ("fc3", nn.Linear(64, 10)),
    ]))


def lenet5_modular(activation_name="relu"):
    return ModularCNN(1, [6, 16], [120, 84], 10, image_size=28,
                      activation_name=activation_name, convs=2, paddings=[1, 0])


def cifarnet_modular(enc_sizes=(16, 32, 64), activation_name="relu"):
    return ModularCNN(3, list(enc_sizes), [256, 64], 10, image_size=32,
                      activation_name=activation_name, dec_activation="tanh",
                      input_norm=True, dec_batch_norm=True)


MODELS = {
    "mnist": {
        "classic": ("nn.Module (ПР №10)", LeNet5),
        "sequential": ("nn.Sequential", lenet5_sequential),
        "modular": ("модулі: relu", lenet5_modular),
        "modular_lrelu": ("модулі: lrelu", lambda: lenet5_modular("lrelu")),
    },
    "cifar": {
        "classic": ("nn.Module (ПР №10)", CifarNet),
        "sequential": ("nn.Sequential", cifarnet_sequential),
        "modular": ("модулі: relu", cifarnet_modular),
        "modular_lrelu": ("модулі: lrelu",
                          lambda: cifarnet_modular(activation_name="lrelu")),
        "modular_wide": ("модулі: 32-64-128",
                         lambda: cifarnet_modular(enc_sizes=(32, 64, 128))),
    },
}


def build(dataset, key):
    return MODELS[dataset][key][1]()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def copy_weights(source, target):
    source_state = list(source.state_dict().values())
    target_keys = list(target.state_dict().keys())
    if len(source_state) != len(target_keys):
        raise ValueError("the two models do not have the same tensors")
    target.load_state_dict(dict(zip(target_keys, source_state)))

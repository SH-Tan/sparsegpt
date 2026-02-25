import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=None, name=""):
    if layers is None:
        layers = (nn.Conv2d, nn.Linear)

    res = {}

    if isinstance(module, layers):
        res[name] = module
        return res

    for name1, child in module.named_children():
        child_name = f"{name}.{name1}" if name else name1
        res.update(find_layers(child, layers=layers, name=child_name))

    return res

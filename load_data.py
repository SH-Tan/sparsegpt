import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset,DataLoader,Dataset
import torchvision
import torch.optim as optim
from Models import lenet5_relu
from Models import lenet5_tanh
from Models import vgg9_relu
from Models import vgg9_tanh


def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    # transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)


class TransformedTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.images, self.labels = tensors
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

def load_dataset_from_disk(path, batch_size=128, shuffle=True, transform=None):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TransformedTensorDataset((images, labels), transform=transform)
    return dataset


def load_dataset_from_disk_mnist(path, batch_size=128, shuffle=True):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TensorDataset(images, labels)
    return dataset


def dataloader(dataset, batch_size, train, workers, length=None, name = ""):
    # Dataset
    if dataset == 'mnist':
        if name == 'test':
            dataset = datasets.MNIST('Data/', train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        else:
            dataset = load_dataset_from_disk("Data/" + name, batch_size=batch_size)
    if dataset == 'cifar10':
        if name == 'test':
            transform_test = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = datasets.CIFAR10('Data/', train=train, download=True, transform=transform_test)
        else:
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
            dataset = load_dataset_from_disk("Data/" + name, batch_size=batch_size)
    if dataset == 'cifar100':
        if name == 'test':
            transform_test = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = datasets.CIFAR100('Data/', train=train, download=True, transform=transform_test)
        else:
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
            dataset = load_dataset_from_disk("Data/" + name, batch_size=batch_size)
        
        
    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             **kwargs)

    return dataloader, dataset


def model(model_architecture, model_class):
    default_models = {
        'lenet5_relu': lenet5_relu.LeNet5,
        'lenet5_tanh': lenet5_tanh.LeNet5,
        "vgg9_relu": vgg9_relu.VGG,
        "vgg9_tanh": vgg9_tanh.VGG,
    }
    models = {
        'default' : default_models,
    }
    if model_class == 'imagenet':
        print("WARNING: ImageNet models do not implement `dense_classifier`.")
    return models[model_class][model_architecture]


def optimizer(optimizer):
    optimizers = {
        'adam' : (optim.Adam, {}),
        'sgd' : (optim.SGD, {}),
        'momentum' : (optim.SGD, {'momentum' : 0.9, 'nesterov' : True}),
        'rms' : (optim.RMSprop, {})
    }
    return optimizers[optimizer]


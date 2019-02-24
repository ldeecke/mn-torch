import torch, torchvision


def CIFAR10(data_path, batch_size, train=True, img_size=32, shuffle=True):
    '''
    Data loader that gives access to images from CIFAR10.
    args:
        data_path:      str
        batch_size:     int
        train:          bool
        img_size:       int
        shuffle:        bool
    returns:
        data_loader:    torch.utils.data.DataLoader
        img_size:       int
        len(dataset):   int
        num_classes:    int
    '''
    num_classes = 10
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ])

    if train:
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform_train)
    else:
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform_test)

    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=2), img_size, len(dataset), num_classes


def CIFAR100(data_path, batch_size, train=True, img_size=32, shuffle=True):
    '''
    Data loader that gives access to images from CIFAR100.
    args:
        data_path:      str
        batch_size:     int
        train:          bool
        img_size:       int
        shuffle:        bool
    returns:
        data_loader:    torch.utils.data.DataLoader
        img_size:       int
        len(dataset):   int
        num_classes:    int
    '''
    num_classes = 100
    transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
            ])
    transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
            ])

    if train:
        dataset = torchvision.datasets.CIFAR100(root=data_path, train=train, download=True, transform=transform_train)
    else:
        dataset = torchvision.datasets.CIFAR100(root=data_path, train=train, download=True, transform=transform_test)

    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=2), img_size, len(dataset), num_classes

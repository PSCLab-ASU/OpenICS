import torchvision
from torchvision import datasets, transforms
def generate_dataset(dataset,input_channel,input_width,input_height,stage, specifics):
    if stage == "testing":
        if dataset == 'bsd500_patch':
            test_dataset = datasets.ImageFolder(root='./data', train= False, download=False,
                                                transform=transforms.Compose([
                                                    transforms.Resize(input_width),
                                                    transforms.CenterCrop(input_width),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))
        elif dataset == 'cifar10':
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
        elif dataset == 'mnist':
            test_dataset = datasets.MNIST('./data', train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
        return test_dataset
    elif stage == "training":
        if dataset == 'lsun':
            train_dataset = datasets.LSUN(root='./data', classes=['bedroom_train'],
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
        elif dataset == 'mnist':
            train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
            val_dataset = datasets.MNIST(root='./data/val', train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
        elif dataset == 'bsd500':
            train_dataset = datasets.ImageFolder(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(input_width),
                                                    transforms.CenterCrop(input_width),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))
            val_dataset = datasets.ImageFolder(root='./data/val',
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_widthe),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
        elif dataset == 'bsd500_patch':
            train_dataset = datasets.ImageFolder(root='./data/val',
                                                transform=transforms.Compose([
                                                    transforms.Resize(input_width),
                                                    transforms.CenterCrop(input_width),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))

            val_dataset = datasets.ImageFolder(root='./data/val',
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
        elif dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root='./data',train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))

            val_dataset = datasets.CIFAR10(root='./data',train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))

        return (train_dataset, val_dataset)

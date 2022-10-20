import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_cifar100_dataloaders(train_batch_size, test_batch_size, num_workers):
    CIFAR_train = torchvision.datasets.CIFAR100("./data/CIFAR_train", download=True, train=True, transform=transforms.ToTensor())
    CIFAR_val_test = torchvision.datasets.CIFAR100("./data/CIFAR_test", download=True, train=False, transform=transforms.ToTensor())
    val_size = int(len(CIFAR_val_test) * 0.5)
    CIFAR_val, CIFAR_test = random_split(CIFAR_val_test, [val_size, len(CIFAR_val_test) - val_size])

    train_loader = DataLoader(CIFAR_train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(CIFAR_val, batch_size=50, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(CIFAR_test, batch_size=50, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(128, 50, 4)
    print("Got the data loaders!")
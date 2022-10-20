import fire
import torch
from mixer import MLPMixer
from vit import VisionTransformer
from mae import MAEVisionTransformer

def train(model="mixer", dataset="cifar100", patch_size=4, epochs=100, batch_size=128, lr=1e-3, wd=1e-6, num_workers=4):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataloaders
    if dataset == "cifar100":
        train_loader, val_loader, test_loader = get_cifar100_dataloaders(batch_size, batch_size, num_workers)
        img_size = 32
    else:
        raise ValueError("Dataset not supported")
    
    # Hyperparameters
    hparams = {

    }

    # Get model
    if model == "mixer":
        model = MLPMixer(**hparams)
    elif model == "vit":
        model = VisionTransformer(**hparams)
    elif model == "mae":
        model = MAEVisionTransformer(**hparams)
    else: 
        raise ValueError("Model not supported")
    
    # Get optimizer & loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(epochs):
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # Forward pass


if __name__ == "__main__":
    fire.Fire(train)
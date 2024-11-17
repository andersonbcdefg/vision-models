import fire
import torch
import torch.nn as nn
from einops import rearrange
from data import get_cifar100_dataloaders
from mae import MAEVisionTransformer

def train_mae( 
    dataset="cifar100", 
    patch_size=4,
    mask_prob=0.75, 
    enc_depth=16, 
    enc_n_heads=8, 
    enc_embed_dim=512, 
    enc_d_k=64, 
    enc_ffn_hidden_dim=2048,
    dec_depth=3,
    dec_n_heads=8,
    dec_embed_dim=512,
    dec_d_k=64,
    dec_ffn_hidden_dim=2048,
    pretrain_head_hidden_dim=2048,
    dropout=0.1,
    max_steps=25000,
    max_epochs=100, 
    batch_size=128, 
    lr=1e-3, 
    wd=1e-6, 
    num_workers=4
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Get dataloaders
    if dataset == "cifar100":
        train_loader, val_loader, test_loader = get_cifar100_dataloaders(batch_size, batch_size, num_workers)
        img_size = 32
        n_channels = 3
    else:
        raise ValueError("Dataset not supported")
    
    assert img_size % patch_size == 0, "Image must divide evenly into patches."
    seq_len = (img_size // patch_size) ** 2
    print("Got the data loaders!")

    # Hyperparameters
    kwargs = {
        "img_size": img_size,
        "patch_size": patch_size, 
        "n_channels": n_channels, 
        "mask_prob": mask_prob,
        "enc_depth": enc_depth, 
        "enc_n_heads": enc_n_heads,
        "enc_embed_dim" : enc_embed_dim,
        "enc_d_k": enc_d_k,
        "enc_ffn_hidden_dim": enc_ffn_hidden_dim,
        "dec_depth": dec_depth,
        "dec_n_heads": dec_n_heads,
        "dec_embed_dim": dec_embed_dim,
        "dec_d_k": dec_d_k,
        "dec_ffn_hidden_dim": dec_ffn_hidden_dim,
        "pretrain_head_hidden_dim": pretrain_head_hidden_dim,
        "dropout": dropout
    }

    # Get model
    model = MAEVisionTransformer(**kwargs).to(device)
    print("Got the model!")

    # Get optimizer & loss function
    print("Optimizing with Adam, learning rate: ", lr, "weight decay: ", wd)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Train
    print("Training for ", max_steps, " steps or ", max_epochs, " epochs")
    model.train()
    epochs = 0
    steps = 0
    running_loss = 0.0
    while True:
        epochs += 1
        for it, batch in enumerate(train_loader):
            steps += 1
            X = batch[0].to(device)
            optimizer.zero_grad()
            masked_idxs, out_patches = model(X)

            # Only use masked patches for loss
            preds = out_patches[:, masked_idxs, :]
            original_patches = rearrange(X, "b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)", h2=patch_size, w2=patch_size)
            targets = original_patches[:, masked_idxs, :]
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % 100 == 0:
                print(f"STEP {steps} | TRAIN LOSS: {running_loss / 100:.3f}")
                running_loss = 0.0
        if steps >= max_steps or epochs >= max_epochs:
            break
        scheduler.step()


if __name__ == "__main__":
    fire.Fire(train_mae)
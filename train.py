import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import GoProDataset
from model import BongNet


EPOCHS = 1000
L1_EPOCHS = 330
BATCH_SIZE = 2
PATCH_SIZE = 256
LR_L1 = 1e-4
LR_LOG_MSE = 3e-5
MIX_P = 0.5
WEIGHT_SAVE_EPOCH = 5
SEED = 1234
VALIDATION_SCENES = {"GOPR0396", "GOPR0410", "GOPR0854", "GOPR0862", "GOPR0869"}


def l1_loss(prediction, target):
    return (prediction - target).abs().mean()


def log_mse_loss(prediction, target):
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    return torch.log(mse + 1e-8).mean()


def main():
    parser = argparse.ArgumentParser(description="Train BongNet with the competition split.")
    parser.add_argument("--data-root", type=Path, default=Path("GOPRO_Large"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    args = parser.parse_args()

    if args.epochs < 1 or args.patch_size < 1:
        parser.error("--epochs and --patch-size must be positive")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    dataset = GoProDataset(
        set_type="train",
        data_type="mix",
        dataset_dir=args.data_root,
        mix_p=MIX_P,
        patch_size=args.patch_size,
        augment=True,
    )
    extra = GoProDataset(set_type="test", data_type="mix", dataset_dir=args.data_root)
    for input_path, sharp_path in extra.pairs:
        scene = sharp_path.parent.parent.name.split("_")[0]
        if scene not in VALIDATION_SCENES:
            dataset.pairs.append((input_path, sharp_path))

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    model = BongNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_L1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Training images: {len(dataset)} | Batch size: {BATCH_SIZE}")

    for epoch in range(1, args.epochs + 1):
        if epoch == L1_EPOCHS + 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_LOG_MSE)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if epoch <= L1_EPOCHS:
            criterion = l1_loss
            loss_name = "L1"
        else:
            criterion = log_mse_loss
            loss_name = "Log-MSE"

        total_loss = 0.0
        for blur, sharp in loader:
            blur = blur.to(device)
            sharp = sharp.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                restored = model(blur)

            loss = criterion(restored.float(), sharp)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * blur.size(0)

        mean_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch}/{args.epochs}] | {loss_name}: {mean_loss:.6f}")
        if epoch % WEIGHT_SAVE_EPOCH == 0:
            torch.save(model.state_dict(), args.output_dir / f"epoch_{epoch:04d}.pt")

    torch.save(model.state_dict(), args.output_dir / "last.pt")


if __name__ == "__main__":
    main()

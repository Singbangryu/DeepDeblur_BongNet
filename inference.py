import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from model import BongNet


def main():
    parser = argparse.ArgumentParser(description="Restore one blurred image with BongNet.")
    parser.add_argument("input", nargs="?", type=Path, default=Path("blur.png"))
    parser.add_argument("--weights", type=Path, default=Path("best.pt"))
    parser.add_argument("--output", type=Path, default=Path("restored.png"))
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input image not found: {args.input}")
    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BongNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    with Image.open(args.input) as image:
        blur = ToTensor()(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        restored = model(blur).squeeze(0).clamp(0, 1).cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ToPILImage()(restored).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

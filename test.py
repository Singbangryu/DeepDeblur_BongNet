import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from model import BongNet


def psnr(prediction, target):
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse)


def main():
    parser = argparse.ArgumentParser(description="Restore and optionally score a directory of images.")
    parser.add_argument("--blur-dir", type=Path, default=Path("test/blur"))
    parser.add_argument("--sharp-dir", type=Path, default=Path("test/sharp"))
    parser.add_argument("--output-dir", type=Path, default=Path("test/restored"))
    parser.add_argument("--weights", type=Path, default=Path("best.pt"))
    parser.add_argument("--blur-pattern", default="blur_*.png")
    parser.add_argument("--sharp-pattern", default="sharp_*.png")
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 for mixed image sizes.")
    parser.add_argument("--inference-only", action="store_true", help="Save outputs without PSNR.")
    args = parser.parse_args()
    evaluate = not args.inference_only

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}")
    if not args.blur_dir.is_dir():
        parser.error(f"Blur directory not found: {args.blur_dir}")
    if evaluate:
        if not args.sharp_dir.is_dir():
            parser.error(f"Sharp directory not found: {args.sharp_dir}")
        if args.blur_pattern.count("*") != 1 or args.sharp_pattern.count("*") != 1:
            parser.error("Evaluation requires one '*' in each filename pattern")
        blur_prefix, blur_suffix = args.blur_pattern.split("*")

    image_paths = sorted(args.blur_dir.glob(args.blur_pattern))
    if not image_paths:
        parser.error(f"No images matched {args.blur_pattern!r} in {args.blur_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BongNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    to_tensor = ToTensor()
    to_image = ToPILImage()
    total_input_psnr = 0.0
    total_restored_psnr = 0.0
    total_gain = 0.0

    with torch.inference_mode():
        for offset in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[offset:offset + args.batch_size]
            blur_batch = []
            sharp_batch = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    blur_batch.append(to_tensor(image.convert("RGB")))

                if evaluate:
                    image_id = image_path.name.removeprefix(blur_prefix).removesuffix(blur_suffix)
                    sharp_path = args.sharp_dir / args.sharp_pattern.replace("*", image_id)
                    with Image.open(sharp_path) as image:
                        sharp_batch.append(to_tensor(image.convert("RGB")))

            blur = torch.stack(blur_batch).to(device)
            restored = model(blur).clamp(0, 1)
            if evaluate:
                sharp = torch.stack(sharp_batch).to(device)
                input_psnr = psnr(blur, sharp)
                restored_psnr = psnr(restored, sharp)
                gains = (restored_psnr - input_psnr).tolist()
                input_scores = input_psnr.tolist()
                restored_scores = restored_psnr.tolist()

            restored = restored.cpu()
            for i, image_path in enumerate(batch_paths):
                to_image(restored[i]).save(args.output_dir / image_path.name)
                status = f"[{offset + i + 1}/{len(image_paths)}] {image_path.name}"
                if evaluate:
                    total_input_psnr += input_scores[i]
                    total_restored_psnr += restored_scores[i]
                    total_gain += gains[i]
                    print(f"{status} | Input {input_scores[i]:.4f} dB | "
                          f"Restored {restored_scores[i]:.4f} dB | Gain {gains[i]:+.4f} dB")
                else:
                    print(f"{status} | Restored")

    if evaluate:
        n = len(image_paths)
        print(f"Mean input PSNR: {total_input_psnr / n:.4f} dB")
        print(f"Mean restored PSNR: {total_restored_psnr / n:.4f} dB")
        print(f"Mean gain: {total_gain / n:+.4f} dB")


if __name__ == "__main__":
    main()
